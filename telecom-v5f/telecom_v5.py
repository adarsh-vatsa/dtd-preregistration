"""Telecom controller v5 ("dstx"): the table/belief picks every move; the controller executes every move whose content is fixed
by the state; the LLM is called only for turns no template covers (off-script questions, unexpected replies).

Executed moves (no LLM call), on top of v4's wide templates (telecom_tracker.dst_next_wide):
  lookups      customer by phone, line details, the line's plan (data limit, refuel price), the customer's bills
  blockers     SIM locked / contract ended -> transfer_to_human_agents, then the hold message (policy)
  suspension   overdue bill -> explain + offer payment request -> send_payment_request on yes -> after the customer accepts,
               check the bill is paid -> resume_line -> ask for a reboot and re-test (policy: Overdue Bill Payment, Line Suspension)
  data limit   usage over the plan limit -> offer refuel (price per GB, max 2 GB) -> confirm price -> refuel_data (policy)
  roaming      customer abroad and line roaming off -> enable_roaming (policy: Data Roaming), plus the phone-side instruction
  phone steps  one templated instruction per step, each sent at most once (+ one re-ask), in the order chosen by `decide`:
                 mdp     static DP order (bounded/order_dp.json): v4's table on the top-guess state
                 belief  posterior over the task's fault set (joint fault mix of the TRAIN split), updated by Bayes from each
                         step's reported outcome (found / not found / unclear) and the still-failing symptom; next step =
                         highest posterior probability that its fault is present and unfixed (greedy p/c, unit cost); equals
                         QMDP here because every step both tests and fixes its fault
                 pomdp   belief + two-step lookahead on the expected number of remaining turns (checks whether testing a
                         likelier-but-correlated step first shortens the expected path)
  closing      symptom reported fixed -> closing message

v5f changes (diagnosed on the spent draws 1-2 and train-split logs; no task-id logic):
  evidence     every customer message is scanned, per sentence, for a reported fault (status-bar icons, network-status lines,
               MMSC field, app-permission list, Wi-Fi Calling, network mode, VPN, data roaming), whichever check is pending;
               all faults in a message are queued (v5e kept only the first). A sentence counts unless it reports the customer's
               own fix ("I've turned it off"); v5e's message-wide FIX_DONE gate also fired on questions ("Should I reset it?")
               and state descriptions ("is turned on") and suppressed most MMSC reports.
  fixes        a fault with evidence gets its fix template (now also app permissions with the missing names, phone data
               roaming, and a restart when an APN reset is waiting for a reboot); at most two tries per fault.
  re-asks      a reply that defers without a result ("Let me check the airplane mode first") or refuses a two-step bundle
               ("one at a time") gets that step re-asked once; bundling stops after a refusal.
  readouts     when every check has been sent and the problem persists, ask once each for the readouts that reveal the
               remaining faults (network status, MMSC field, messaging-app permissions, status bar, ...) before the LLM.
  success      wider MMS-works detection ("it went through", "it's working now", "I'm able to send").
"""
import collections, itertools, json, math, os, re

import telecom_tracker as T

COUNTRIES = r"(?:france|spain|italy|germany|portugal|greece|mexico|canada|japan|china|india|brazil|europe|asia|the uk|england|ireland|london|paris)"
ABROAD_STRONG = re.compile(r"\babroad\b|overseas|in (?:another|a different|a foreign) country|outside (?:of )?(?:the |my )?(?:country|home (?:country|area|network))|"
                           r"i(?:'m| am) (?:\w+ ){0,3}in " + COUNTRIES + r"|visiting (?:\w+ ){0,3}in " + COUNTRIES)
ABROAD_WEAK = re.compile(r"\btravel(?:l)?ing\b(?! (?:to|back) (?:home|work))|\bon (?:a )?(?:trip|vacation|holiday)\b")
NOT_ABROAD = re.compile(r"\bno travel(?:l)?ing|not (?:currently |really )?(?:abroad|travel(?:l)?ing|overseas)|(?<!not )\bat home\b|"
                        r"haven'?t (?:been|gone) anywhere|\bin my (?:living room|house|home)\b")
YES = re.compile(r"\b(?:yes|yeah|yep|sure|please do|please go ahead|go ahead|ok(?:ay)?|that'?s fine|sounds good|do it|send it|i accept|absolutely|of course)\b", re.I)
NO = re.compile(r"\b(?:no|nope|don'?t|do not|not now|rather not|decline|cancel)\b", re.I)
GB = re.compile(r"(\d+(?:\.\d+)?)\s*(?:gb|gigabytes?|gigs?)\b|\b(one|two|1|2)\s*(?:gb|gigabytes?|gigs?)\b", re.I)
WORDNUM = {"one": 1.0, "two": 2.0}
PRIORITY_CUES = {"airplane": r"airplane (?:mode )?(?:symbol|icon|is on|was on|turned on|showing|still)|airplane symbol",
                 "wifi_calling": r"wi-?fi calling (?:is |was )?(?:on|enabled)", "data_saver": r"data saver (?:is |was )?(?:on|enabled)",
                 "vpn": r"\bvpn\b (?:is |was )?(?:on|connected)", "mobile_data": r"mobile data (?:is |was )?(?:off|disabled)"}
BUNDLE = os.environ.get("DTD_BUNDLE", "1") == "1"
# v5f: fault evidence in a customer message, matched per sentence: step -> (pattern, guard). The patterns follow the readouts the
# phone tools print (tau2 user_tools) and their paraphrases; a guard hit in the same sentence (the setting is fine / off) cancels.
_NEG_SEE = r"don'?t (?:see|think)|not (?:on|see|showing)|isn'?t (?:on|showing)"
_GONE = r"\b(?:is |are )?(?:gone|no longer (?:there|showing|on)|disappeared)\b|\bno more\b"
_OFF = r"(?:is|was|'s|are|turned|switched|now|currently|already|it'?s|says|shows) (?:turned |switched |set to )?(?:off|disabled)\b|:\s*off\b"
EVIDENCE = {
    "airplane": (r"airplane(?: mode)? (?:symbol|icon)|airplane mode: on\b|airplane mode\b[^.!?]{0,30}\b(?:is|was|'s|still)\b[^.!?]{0,15}\b(?:on|enabled|active)\b",
                 _OFF + "|" + _GONE + r"|\bno airplane|not (?:on|showing|see)|don'?t see|isn'?t (?:on|showing)"),
    "mobile_data": (r"\bdata disabled\b|mobile data enabled:? no\b|mobile data\b[^.!?,]{0,40}\b(?:off|disabled)\b",
                    r"mobile data (?:is |was )?(?:now |already |still )?(?:on|enabled)\b|\bdata enabled\b|no signal|no service|"
                    r"(?:turned|switched) on (?:the |my )?mobile data|enabled (?:the |my )?mobile data|" + _GONE),
    "sim": (r"\bno sim\b(?! (?:issue|problem))|\bsim\b[^.!?,]{0,40}\b(?:missing|not detected|not inserted|not found|absent)\b",
            r"not missing|isn'?t missing|\bsim (?:card )?(?:status )?(?:is|was|shows|says)(?: now)? (?:active|fine|working|detected)\b"),
    "wifi_calling": (r"wi-?fi calling\b[^.!?,]{0,40}\b(?:on|enabled|active)\b", _OFF + "|" + _NEG_SEE),
    "data_saver": (r"data saver\b[^.!?,]{0,40}\b(?:on|enabled|active|icon|symbol)\b", _OFF + "|" + _NEG_SEE + "|" + _GONE),
    "vpn": (r"\bvpn\b[^.!?,]{0,40}\b(?:on|connected|active|enabled)\b",
            r"\bno vpn\b|not connected|isn'?t connected|disconnected|not (?:on|using|active|enabled)|no active vpn|" + _OFF + "|" + _NEG_SEE + "|" + _GONE),
    "netpref": (r"\b[23]g[_ ]only\b|network mode\b[^.!?]{0,40}\b[23]g\b|(?:set to|connected to|shows?|showing|\bon) (?:an? |only )?[23]g\b",
                r"\b4g\b|\b5g\b|\blte\b|(?:\bnot\b|n't)[^.!?,]{0,20}\b[23]g\b"),
    "apn_mms": (r"mmsc\b[^.!?]{0,80}(?:not set|missing|empty|blank|\bnone\b|isn'?t set|nothing)|(?:not set|missing|empty|blank)[^.!?]{0,40}\bmmsc",
                r"https?:|mms\.carrier|no longer|now set|\bis set\b|not missing|isn'?t missing|doesn'?t look like it'?s missing"),
    "roaming": (r"data roaming\b[^.!?,]{0,40}\b(?:off|disabled|not enabled)\b|data roaming enabled:? no\b",
                r"(?:is|was) (?:now |already )?(?:on|enabled)\b|now on\b"),
}
# a sentence in which the customer reports fixing a setting is not evidence of a fault (questions and plans are)
SETTING_FIXED = re.compile(r"\bi(?:'ve| have| just)? (?:now |already )?(?:turned|switched|toggled|changed|disconnected|re-?seated|"
                           r"re-?inserted|reset|granted|disabled|enabled|put (?:it|the sim(?: card)?) back)\b")
STEP_WORD = {"airplane": r"airplane", "mobile_data": r"mobile data|\bdata\b(?! (?:roaming|saver|usage|limit|plan))", "sim": r"\bsim\b", "wifi_calling": r"wi-?fi calling",
             "data_saver": r"data saver", "vpn": r"\bvpn\b", "netpref": r"network mode|preference|\b[2345]g\b", "apn_mms": r"\bapn\b|mmsc",
             "roaming": r"roaming", "app_perm": r"permission|\bsms\b|storage"}
CLEARED = {"airplane": r"airplane[^.!?]{0,30}\b(?:off|disabled)\b", "mobile_data": r"mobile data[^.!?]{0,30}\b(?:on|enabled)\b|\bdata enabled\b",
           "sim": r"(?<!no )\bsim (?:card )?(?:status )?(?:is|was|shows|says|now)(?: now| back)?(?: as)? (?:active|working|detected)\b", "wifi_calling": r"wi-?fi calling[^.!?]{0,30}\b(?:off|disabled)\b",
           "data_saver": r"data saver[^.!?]{0,30}\b(?:off|disabled)\b", "vpn": r"\bvpn\b[^.!?]{0,30}\b(?:off|disconnected|not connected)\b|\bno vpn\b",
           "netpref": r"(?:set to|changed (?:it )?to|now|preference is|mode is)[^.!?]{0,20}\b(?:4g|5g|lte)\b(?![^.!?]{0,20}\b[23]g\b)", "apn_mms": r"mmsc[^.!?]{0,60}(?:https?:|mms\.)", "roaming": r"data roaming[^.!?]{0,30}\b(?:on|enabled)\b"}
APN_REBOOT_PENDING = re.compile(r"reset (?:at|after|on|upon|when i) (?:the next )?re(?:boot|start)|takes? effect (?:at|after|on) (?:a |the )?re(?:boot|start)")
REBOOTED = re.compile(r"\b(?:restarted|rebooted)\b|turned (?:it|my phone|the phone) (?:off and (?:back )?on|back on)|phone (?:is |has )?(?:back on|restarted)")
APP_NOT_FOUND = re.compile(r"(?:app|messages?|messaging)\b[^.!?]{0,30}\b(?:not found|isn'?t found|wasn'?t found)|can'?t find (?:the |my )?(?:messages?|messaging) app")
DEFER_NO_RESULT = re.compile(r"\b(?:let me|i'?ll|i will|i'?m going to|going to|one moment|give me a (?:second|moment|minute))\b")
RESULT_WORD = re.compile(r"\b(?:says|said|shows?|showing|shown|reads|listed|lists|is (?:on|off|enabled|disabled|set|active|missing|connected|turned)|"
                         r"was (?:on|off|enabled|disabled|turned)|already|not set|missing|granted|enabled|disabled|active|connected|excellent|"
                         r"good|fair|poor|no signal|\d)\b")
BUNDLE_REFUSED = re.compile(r"one (?:action|thing|step|task) at a time|multiple (?:actions|things|steps) at once|"
                            r"can(?:'t|not) (?:do|perform) (?:two|both|multiple|more than one)|which one (?:to|should i) (?:do|check) first")
MMS_WORKS = re.compile(r"\b(?:went|goes|going|gone) through\b|\bit(?:'s| is)? (?:now )?working\b|\bit worked\b|"
                       r"\b(?:was|am|i'm|i am) (?:now )?able to send\b|\bcan (?:now )?send\b|\b(?:sent|sending) (?:successfully|fine)\b|successfully sent")
MMS_NOT = re.compile(r"\b(?:not|never|unable|cannot|still|fail\w*|see if|whether|let me|if it|try(?:ing)? (?:again|to))\b|n't\b")
OTHER_SETTING = re.compile(r"\bsim\b|signal|\bdata\b|wi-?fi|airplane|\bvpn\b|roaming|\bapn\b|says it")   # "the SIM says it's working now"
MMS_WORD = re.compile(r"\bmms\b|picture|photo|message|text")
_SENT = re.compile(r"(?<=[.!?])[\"'”’)]*\s+|\n+")
READOUTS = [   # v5f: asked once each when every check has been sent and the problem persists (one action per message)
    ("network_status", ("service", "data", "mms"), "Please open your phone's network status and read me each line exactly as it appears, "
                                                   "including Airplane Mode, SIM Card Status, Mobile Data Enabled and Data Roaming Enabled."),
    ("apn", ("mms",), "Please open your APN settings and read me exactly what the MMSC URL field says."),
    ("perm", ("mms",), "Please check the permissions of the messaging app (the app is called \"messaging\") and read me the exact list of "
                       "permissions it has."),
    ("status_bar", ("service", "data", "mms"), "Please look at the status bar at the top of your screen and read me exactly what it shows, "
                                               "every icon and word."),
    ("wifi_calling", ("mms",), "Please check Wi-Fi Calling and tell me exactly whether it is on or off."),
    ("netpref", ("data", "mms"), "Please check your network mode preference and tell me exactly which mode it is set to."),
    ("vpn", ("data",), "Please check your VPN status and tell me exactly what it says."),
    ("data_saver", ("data",), "Please check Data Saver mode and tell me exactly whether it is on or off."),
]


def missing_perms(sent):
    """SMS / storage permissions a sentence reports as missing from the messaging app (a listed set that lacks them, or a negation)."""
    if re.search(r"\bi(?:'ve| have)? (?:granted|gave|added|enabled)\b|has been granted|now (?:has|granted)|granted to", sent):
        return set()
    out = set()
    m = re.search(r"permissions? (?:for|to)(?: access)?:?\s*([^.!?]*)", sent)
    if m and re.search(r"\b(?:camera|location|contacts|microphone|phone|notifications|storage|sms)\b", m.group(1)):
        out |= {x for x in ("sms", "storage") if not re.search(r"\b" + x + r"\b", m.group(1))}
    for x in ("sms", "storage"):
        if re.search(r"(?:\bno\b|missing|without|lacks?|doesn'?t have|does not have|don'?t see|isn'?t|\bnot\b)[^.!?]{0,40}\b" + x + r"\b", sent):
            out.add(x)
    if re.search(r"\bno permissions\b|has no permission|not have any permission", sent):
        out |= {"sms", "storage"}
    return out


GOOD_STATE = {"mobile_data": r"on|enabled", "roaming": r"on|enabled", "airplane": r"off|disabled", "wifi_calling": r"off|disabled",
              "data_saver": r"off|disabled", "vpn": r"off|disconnected|disabled", "app_perm": r"granted"}
_CLAUSE = re.compile(r"(?<=[.!?])[\"'\u201d\u2019)]*\s+|\n+|;\s*|,?\s+(?:but|however|though|and now)\b\s*|\s+-\s+")
HYPOTHETICAL = re.compile(r"\b(?:see|check|look|find out|know|ask) (?:if|whether)\b|\bwhether\b|\bif (?:it'?s|it is|its|that'?s|that is|the|my|they|there)\b|\bsuch as\b|"
                          r"(?:was|were) (?:the|a) (?:issue|problem|culprit|cause)|turns? out|turned out|(?:was|were) causing|"
                          r"\brecap\b|the (?:issues?|problems?) (?:was|were)|(?:it was|that was) (?:a |quite a )?(?:bit of a )?journey")   # + hindsight
MODAL = re.compile(r"\b(?:should|shall|can|could|will|would|want me to|let me|i'?ll|going to|need to|have to|try to|supposed to)\b")


def evidence_events(text):
    """Ordered ('add'|'clear', step) events for one customer message, clause by clause, plus the permissions reported missing.
    'add': the clause reports the step's fault; 'clear': it reports the setting fine, or the customer's own fix of it
    ('I changed it to 4G/5G preferred' clears the network mode named in the clause before). Questions, plans and hypotheticals
    ('Should I reset it?', 'let me see if Wi-Fi Calling is on') are not fixes and not evidence."""
    ev, perms, prev = [], set(), []
    for cl in _CLAUSE.split(T.norm(text)):
        if not cl or not cl.strip():
            continue
        fixed = bool(SETTING_FIXED.search(cl)) and not cl.rstrip().endswith("?") and not MODAL.search(cl)
        added = []
        if not fixed and not HYPOTHETICAL.search(cl):
            hits = []
            for step, (pat, guard) in EVIDENCE.items():
                m = re.search(pat, cl)
                if m and not (guard and re.search(guard, cl) and not (step == "netpref" and re.search(r"\b[23]g[_ ]only\b", cl))):
                    hits.append((m.start(), step))
            added = [x for _, x in sorted(hits)]
            p = missing_perms(cl)
            if p:
                perms |= p; added.append("app_perm")
            ev += [("add", x) for x in added]
        for step, pat in CLEARED.items():
            if re.search(pat, cl) or (re.search(_GONE, cl) and re.search(STEP_WORD.get(step, r"$^"), cl) and step != "app_perm"):
                ev.append(("clear", step))
        if fixed:
            ev += [("clear", step) for step, word in STEP_WORD.items() if re.search(word, cl)]
            if re.search(r"\b(?:it|that|them|those)\b", cl):
                ev += [("clear", step) for step in prev]
        m = re.search(r"\b(?:it'?s|it is|it was|that'?s) (?:now |already )?(on|off|enabled|disabled|granted|connected|disconnected)\b", cl)
        if m and not added:
            ev += [("clear", step) for step in prev if re.search(GOOD_STATE.get(step, r"$^"), m.group(1))]
        prev = added or prev
    return ev, perms


def fault_evidence(text):
    """(steps, perms): steps whose fault this message reports and does not also report fixed, in order (used offline)."""
    ev, perms = evidence_events(text)
    out = []
    for op, step in ev:
        if op == "add" and step not in out:
            out.append(step)
        elif op == "clear" and step in out:
            out.remove(step)
    return out, perms


FIX_TEMPLATE = {"netpref": "Thanks. Please change your network mode preference to 4G/5G preferred, {retest}.",
                "airplane": "Thanks. Please turn Airplane Mode off, {retest}.", "wifi_calling": "Thanks. Please turn Wi-Fi Calling off, {retest}.",
                "data_saver": "Thanks. Please turn Data Saver mode off, {retest}.", "vpn": "Thanks. Please disconnect the VPN, {retest}.",
                "mobile_data": "Thanks. Please turn Mobile Data on, {retest}.", "sim": "Thanks. Please take the SIM card out and re-insert it (re-seat it), {retest}.",
                "apn_mms": "Thanks. Please reset your APN settings to default, then restart your phone (the reset only takes effect after "
                           "the restart; restarting does not delete your photos or messages), {retest}.",
                "roaming": "Thanks. Please turn Data Roaming on in your phone's settings, {retest}.",
                "apn_reboot": "Thanks. The APN reset only takes effect after a restart, so please restart your phone now (restarting does not "
                              "delete your photos or messages), {retest}.",
                "app_perm": "Thanks. Please open the permissions of the messaging app (the app is called \"messaging\") and grant the {need} "
                            "permission, {retest}.",
                "app_name": "The messaging app on your phone is called \"messaging\". Please check the permissions of the \"messaging\" app and "
                            "tell me exactly which permissions it lists."}
ASK_TRAVEL = "Before we continue, one quick question: are you currently travelling outside your home country (abroad)?"
PAID_DONE = re.compile(r"accept|paid|made the payment|payment (?:is |has been )?(?:done|made|complete|sent)|went through|completed|done", re.I)

# step -> faults it fixes (task-generator atoms)
STEP_FAULTS = {"airplane": {"airplane_mode_on"}, "sim": {"unseat_sim_card"}, "apn_reset": {"break_apn_settings"},
               "roaming": {"user_abroad_roaming_disabled_on", "user_abroad_roaming_disabled_off", "user_abroad_roaming_enabled_off"},
               "mobile_data": {"data_mode_off"}, "usage": {"data_usage_exceeded"}, "data_saver": {"data_saver_mode_on"},
               "netpref": {"bad_network_preference"}, "vpn": {"bad_vpn"}, "wifi_calling": {"bad_wifi_calling"},
               "apn_mms": {"break_apn_mms_setting"}, "app_perm": {"break_app_sms_permission", "break_app_storage_permission", "break_app_both_permissions"},
               "reboot": set(), "line": {"overdue_bill_suspension", "contract_end_suspension"}}
FOUND = re.compile(r"\b(?:was|were) (?:on|off|enabled|disabled|missing|wrong|incorrect|not set|set to|turned on|connected|denied|blocked)\b|"
                   r"\bi (?:turned|switched|toggled|changed|reset|re-?seated|reinserted|re-?inserted|granted|disconnected|enabled|disabled|took out)\b|"
                   r"\b(?:it'?s|its|is) now (?:on|off|enabled|disabled|set|granted|connected)\b|\bnow (?:enabled|disabled|granted)\b|"
                   r"\bhave (?:turned|switched|changed|reset|granted|disconnected|enabled|disabled)\b", re.I)
NOT_FOUND = re.compile(r"\balready (?:off|on|enabled|disabled|set|connected|granted|disconnected|turned)\b|\b(?:was|is|looks|seems) "
                       r"(?:fine|correct|ok|okay|normal|right|good)\b|\bno vpn\b|\bnot (?:connected to|using) (?:a |any )?vpn\b|"
                       r"\bnothing (?:changed|to change|was wrong)\b|\b(?:is|was) (?:off|on) already\b", re.I)


def step_outcome(text):
    t = text or ""
    f, n = bool(FOUND.search(t)), bool(NOT_FOUND.search(t))
    return "found" if f and not n else ("not_found" if n and not f else "unclear")


class FaultPrior:
    """Joint distribution of fault sets per intent from the TRAIN split's task ids."""

    def __init__(self, tasks_json, splits_json):
        tasks = json.load(open(tasks_json)); sp = json.load(open(splits_json))
        train = set(sp["train"])
        self.sets = {}
        for t in tasks:
            if train and t["id"] not in train:
                continue
            intent = {"service_issue": "service", "mobile_data_issue": "data", "mms_issue": "mms"}.get(t["id"].split("]")[0][1:])
            faults = frozenset(t["id"].split("]")[1].split("[")[0].split("|"))
            self.sets.setdefault(intent, collections.Counter())[faults] += 1


class Executor:
    def __init__(self, order, decide="mdp", prior=None, eps=0.1):
        self.order, self.decide, self.prior, self.eps = order, decide, prior, eps
        self.st = {}                       # v4 wide-template bookkeeping (sent, reasked, closed, transferred, held)
        self.abroad = None
        self.plan = None; self.plan_asked = False
        self.bills = None; self.bills_asked = False
        self.pay = {"stage": None, "bill": None, "checked": 0}
        self.refuel = {"stage": None, "gb": None}
        self.roam_done = False
        self.after_tool = None             # message to send right after an executed tool returns
        self.obs = {}                      # step -> found / not_found / unclear (customer's report after the step)
        self.last_step = None
        self.last_steps = None             # v5d: steps bundled into the last instruction
        self.perm_followup = None          # v5c: SMS / storage permission not confirmed in the reply -> ask for exactly that
        self.perm_followup_sent = False
        self.fix_pending = None            # v5e: a step whose fault was reported but not fixed
        self.fixes_sent = set()
        self.priority = None               # v5c: a setting the customer says looks wrong is checked next
        self.evidence = []                 # v5f: steps whose fault a customer message reported, not yet acted on (ordered)
        self.perm_missing = set()          # v5f: permissions last reported missing
        self.fix_tries = collections.Counter()
        self.redo, self.redo_sent = [], set()   # v5f: steps answered without a result, re-asked once
        self.result_seen = set()           # v5f: steps for which a reply carried a readout/result
        self.no_bundle = False             # v5f: the customer refused a two-step instruction
        self.readouts_sent = set()
        self.apn_ok_seen = False           # v5f: the customer read out a set MMSC URL (no restart needed after a reset)
        self.asked_travel = False          # v5b: the travel question is asked on its own (a pure information move), because a
        self.awaiting_travel = False       # customer message that also calls a phone tool never reaches the agent

    # ------------------------------------------------------------------ inputs beyond the v4 tracker
    def user(self, tr, text):
        t = T.norm(text)
        if self.awaiting_travel:
            self.awaiting_travel = False
            if self.abroad is None and re.match(r"^\W*(?:yes|yeah|yep|i am|i'm|correct)\b", t):
                self.abroad = True
            elif self.abroad is None and re.match(r"^\W*(?:no|nope|not)\b", t):
                self.abroad = False; tr.handled.add("roaming"); tr.unanswered.discard("roaming")
        strong, weak, neg = ABROAD_STRONG.search(t), ABROAD_WEAK.search(t), NOT_ABROAD.search(t)
        if strong or (weak and not neg):          # a specific cue (abroad, in France) wins; a bare "travelling" loses to "at home"
            self.abroad = True
        elif neg:
            self.abroad = False
            if "roaming" in self.st.get("sent", set()):        # asked about travel; not abroad = roaming step answered
                tr.handled.add("roaming"); tr.unanswered.discard("roaming")
        if self.last_step and self.last_step in tr.handled and self.last_step not in self.obs:
            self.obs[self.last_step] = step_outcome(text)
        # (v5c/v5d permission follow-up parser removed in v5f: superseded by missing_perms, which needs an explicit list or negation)
        # v5f: evidence from every message, clause by clause and in order (a later clause may report the fix)
        asked = self.last_steps or ([self.last_step] if self.last_step else [])
        sents = _SENT.split(t)
        ev, perms = evidence_events(text)
        if perms:
            self.perm_missing = perms
        if re.search(r"mmsc[^.!?]{0,60}(?:https?:|mms\.)", t):
            self.apn_ok_seen = True
        reset_done = any(re.search(r"\breset\b", x) and SETTING_FIXED.search(x) and not MODAL.search(x) and not x.rstrip().endswith("?") for x in sents)
        if (APN_REBOOT_PENDING.search(t) or reset_done) and not REBOOTED.search(t) and not self.apn_ok_seen:
            ev.append(("add", "apn_reboot"))
        if REBOOTED.search(t):
            ev.append(("clear", "apn_reboot"))
        if APP_NOT_FOUND.search(t) and "app_name" not in self.fix_tries:
            ev.append(("add", "app_name"))
        for op, step in ev:
            if op == "clear" and step in self.evidence:
                self.evidence.remove(step)
            elif op == "add" and step not in self.evidence and not (step == "roaming" and self.abroad is False):
                self.evidence.append(step)
        for step in T.STEPS:
            if any(re.search(STEP_WORD.get(step, T.REPLY_KW[step]), x) and RESULT_WORD.search(x) for x in sents):
                self.result_seen.add(step)
        if asked and (BUNDLE_REFUSED.search(t) or (DEFER_NO_RESULT.search(t) and not RESULT_WORD.search(t))):
            self.no_bundle = self.no_bundle or bool(BUNDLE_REFUSED.search(t))
            for step in asked:                       # deferred or refused without a result: re-ask once (v5f)
                if step in T.STEP_TEMPLATE and step not in self.redo and step not in self.redo_sent and step not in self.result_seen:
                    self.redo.append(step)
        if tr.intent == "mms" and tr.symptom != "ok":    # v5f: wider MMS-works detection; the first sentence with an outcome decides
            for x in sents:
                if MMS_WORKS.search(x) and not MMS_NOT.search(x) and not (OTHER_SETTING.search(x) and not MMS_WORD.search(x)):
                    tr.symptom = "ok"; break
                if T.MMS_FAIL.search(x):
                    break
        for step, cue in PRIORITY_CUES.items():
            if step not in self.st.get("sent", set()) and step not in tr.handled and re.search(cue, t):
                self.priority = step

    def tool(self, tr, name, args, result):
        try:
            r = json.loads(result) if isinstance(result, str) else result
        except Exception:
            r = None
        if name == "get_details_by_id" and isinstance(r, dict):
            if r.get("plan_id") and "data_limit_gb" in r:
                self.plan = r
            if r.get("bill_id") and self.pay["bill"] and r["bill_id"] == self.pay["bill"].get("bill_id"):
                self.pay["bill"] = r
        if name == "get_bills_for_customer" and isinstance(r, list):
            self.bills = r
        if name == "refuel_data":                      # v5e: the LLM may perform the refuel itself; never refuel twice
            self.refuel["stage"] = "done"; tr.usage_exceeded = False
            ln0 = tr.lines_checked.get(tr.target_line)
            if ln0 is not None and isinstance(args, dict):
                try:
                    ln0["data_refueling_gb"] = float(ln0.get("data_refueling_gb", 0) or 0) + float(args.get("gb_amount", 0))
                except Exception:
                    pass
        if name == "send_payment_request" and self.pay["stage"] in (None, "offered"):
            self.pay["stage"] = "requested"
        if name == "resume_line":
            self.pay["stage"] = "resumed"
            ln1 = tr.lines_checked.get(tr.target_line)
            if ln1 is not None:
                ln1["status"] = "Active"
        if name == "enable_roaming":
            self.roam_done = True
            ln = tr.lines_checked.get(tr.target_line)
            if ln is not None:
                ln["roaming_enabled"] = True

    # ------------------------------------------------------------------ helpers
    def line(self, tr):
        return tr.lines_checked.get(tr.target_line) or {}

    def over_limit(self, tr):
        ln = self.line(tr)
        if self.plan and ln:
            try:
                return float(ln.get("data_used_gb", 0)) >= float(self.plan["data_limit_gb"]) + float(ln.get("data_refueling_gb", 0) or 0)
            except Exception:
                return False
        return tr.usage_exceeded

    def overdue_bills(self):
        return [b for b in (self.bills or []) if isinstance(b, dict) and str(b.get("status", "")).lower() == "overdue"]

    def retest(self, tr):
        return T.RETEST[tr.intent or "service"]

    def retest_now(self, tr):
        return {"service": "Please check your status bar and tell me what it shows.",
                "data": "Please run a speed test and tell me the result.",
                "mms": "Please try sending a picture message (MMS) again and tell me whether it goes through."}[tr.intent or "service"]

    # ------------------------------------------------------------------ v5f: evidence-driven fixes and readouts
    def next_fix(self, tr):
        while self.evidence:
            step = self.evidence.pop(0)
            if self.fix_tries[step] >= 2 or (step == "app_perm" and not self.perm_missing):
                continue
            if step == "roaming" and self.abroad is False:
                continue
            self.fix_tries[step] += 1; self.fixes_sent.add(step)
            if step == "app_perm":
                need = " and ".join(x.upper() if x == "sms" else x for x in sorted(self.perm_missing, reverse=True))
                tr.pending.add("app_perm"); self.last_step = "app_perm"; self.last_steps = ["app_perm"]
                return ("message", FIX_TEMPLATE["app_perm"].format(need=need, retest=self.retest(tr)), "fix_app_perm")
            if step == "app_name":
                return ("message", FIX_TEMPLATE["app_name"], "fix_app_name")
            if step in T.STEPS:
                tr.pending.add(step)
            return ("message", FIX_TEMPLATE[step].format(retest=self.retest(tr)), "fix_" + step)
        return None

    def next_readout(self, tr):
        for key, intents, text in READOUTS:
            if key not in self.readouts_sent and (tr.intent or "service") in intents:
                self.readouts_sent.add(key)
                return ("message", text, "readout_" + key)
        return None

    # ------------------------------------------------------------------ step order
    def next_step(self, tr):
        cand = [x for x in self.order.get(tr.intent, T.RELEVANT[tr.intent]) if x not in tr.handled and x not in self.st.get("sent", set())]
        cand = [x for x in cand if x != "line"]
        if not cand or self.decide == "mdp" or not self.prior:
            return cand[0] if cand else None
        post = self.posterior(tr)
        def p_present(step, post_):
            fs = STEP_FAULTS.get(step, set())
            return sum(w for F, w in post_.items() if F & fs)
        if self.decide == "belief":
            return max(cand, key=lambda s: (p_present(s, post), -cand.index(s)))
        # pomdp: two-step lookahead on P(fix progress); ties broken by the static order
        best, bv = None, -1
        for s in cand:
            ps = p_present(s, post); v = ps
            rest = [x for x in cand if x != s]
            if rest:
                post_f = self._cond(post, s, True); post_n = self._cond(post, s, False)
                v += ps * max(p_present(x, post_f) for x in rest) + (1 - ps) * max(p_present(x, post_n) for x in rest)
            if v > bv + 1e-12:
                best, bv = s, v
        return best

    def _cond(self, post, step, present):
        fs = STEP_FAULTS.get(step, set()); out = {}
        for F, w in post.items():
            if bool(F & fs) == present:
                out[F - fs] = out.get(F - fs, 0) + w
        z = sum(out.values()) or 1.0
        return {F: w / z for F, w in out.items()}

    def posterior(self, tr):
        """P(remaining unfixed fault set | intent, step reports, symptom still failing). Faults of steps already carried out are
        removed (fixed); a 'found' report raises sets that contained that fault, 'not found' lowers them (misread rate eps)."""
        sets = self.prior.sets.get(tr.intent) or {}
        post = collections.Counter()
        for F, n in sets.items():
            w = float(n)
            for step, o in self.obs.items():
                has = bool(F & STEP_FAULTS.get(step, set()))
                if o == "found":
                    w *= (1 - self.eps) if has else self.eps
                elif o == "not_found":
                    w *= self.eps if has else (1 - self.eps)
            if not w:
                continue
            rem = set(F)
            for step in tr.handled | self.st.get("sent", set()):
                rem -= STEP_FAULTS.get(step, set())
            rem -= {"overdue_bill_suspension", "contract_end_suspension", "lock_sim_card_pin"}
            if tr.symptom != "ok" and not rem:
                w *= self.eps                  # the problem persists, so some fault should remain
            post[frozenset(rem)] += w
        z = sum(post.values()) or 1.0
        return {F: w / z for F, w in post.items()}

    # ------------------------------------------------------------------ the move for this turn
    def move(self, tr, last_user_text, last_was_tool):
        """("message", text, tag) | ("tool", (name, args), tag) | None (LLM)."""
        st = self.st
        if self.after_tool and last_was_tool:
            m, self.after_tool = self.after_tool, None
            return ("message", m, "after_tool")
        if st.get("transferred") and not st.get("held"):
            st["held"] = True; return ("message", T.HOLD, "hold")
        if st.get("held"):
            return None
        if not tr.phone:
            if not st.get("asked_phone"):
                st["asked_phone"] = True; return ("message", T.FIRST_REPLY, "ask_phone")
            return None
        if not tr.customer or not tr.target_line:
            return None
        cid, lid = tr.customer.get("customer_id"), tr.target_line
        b = tr.blocker()
        if b != "none" and not st.get("transferred"):
            st["transferred"] = True
            return ("tool", ("transfer_to_human_agents", {"summary": f"Customer issue ({tr.intent}) cannot be resolved: {b.replace('_', ' ')}."}), "transfer")
        ln = self.line(tr)
        t = (last_user_text or "").strip()
        # ---- suspended line: overdue-bill payment flow
        if str(ln.get("status", "")).lower() == "suspended":
            if self.bills is None:
                if self.bills_asked:
                    return None
                self.bills_asked = True
                return ("tool", ("get_bills_for_customer", {"customer_id": cid}), "bills")
            od = self.overdue_bills()
            p = self.pay
            if p["stage"] is None:
                if not od:
                    return None
                p["bill"] = od[0]; p["stage"] = "offered"
                bl = p["bill"]
                return ("message", f"I found the cause: your line {tr.phone} is suspended because of an overdue bill (bill {bl.get('bill_id')}, "
                                   f"${float(bl.get('total_due', 0)):.2f}, due {bl.get('due_date')}). I can send you a payment request for this bill now. "
                                   f"Would you like me to send it?", "pay_offer")
            if p["stage"] == "offered":
                if YES.search(t) and not re.search(r"\bno\b|don'?t", t, re.I):
                    p["stage"] = "requested"
                    bl = p["bill"]
                    self.after_tool = (f"I've sent you a payment request for ${float(bl.get('total_due', 0)):.2f} for bill {bl.get('bill_id')}. "
                                       f"Please check your payment requests, accept the request, and let me know once you have done that.")
                    return ("tool", ("send_payment_request", {"customer_id": cid, "bill_id": bl.get("bill_id")}), "pay_request")
                return None
            if p["stage"] == "requested":
                p["stage"] = "checking"; p["checked"] += 1
                return ("tool", ("get_details_by_id", {"id": p["bill"].get("bill_id")}), "pay_check")
            if p["stage"] == "checking":
                if str(p["bill"].get("status", "")).lower() == "paid":
                    rest = [x for x in self.overdue_bills() if x.get("bill_id") != p["bill"].get("bill_id")]
                    if rest:                      # more overdue bills: one awaiting-payment request at a time
                        p["bill"] = rest[0]; p["stage"] = "offered"
                        self.bills = [x for x in self.bills if x.get("bill_id") != rest[0].get("bill_id") or True]
                        return None
                    p["stage"] = "resumed"
                    self.after_tool = (f"Thank you. Bill {p['bill'].get('bill_id')} is now paid and I've lifted the suspension, so your line is active "
                                       f"again. Please restart your phone, {self.retest(tr)}.")
                    tr.pending.add("reboot")
                    return ("tool", ("resume_line", {"customer_id": cid, "line_id": lid}), "resume")
                if p["checked"] < 2:
                    p["stage"] = "requested"
                    return ("message", f"I don't see the payment yet. Please check your payment requests, accept the request for "
                                       f"${float(p['bill'].get('total_due', 0)):.2f}, and let me know once it's done.", "pay_remind")
                return None
            return None
        # ---- data intents: plan lookup (limit, refuel price), then refuel flow when over the limit
        if tr.intent in ("data", "mms") and self.plan is None and ln.get("plan_id") and not self.plan_asked:
            self.plan_asked = True
            return ("tool", ("get_details_by_id", {"id": ln["plan_id"]}), "plan")
        if tr.intent in ("data", "mms") and self.over_limit(tr) and self.refuel["stage"] != "done":
            rf = self.refuel; price = float((self.plan or {}).get("data_refueling_price_per_gb", 0) or 0)
            if rf["stage"] is None:
                rf["stage"] = "offered"
                return ("message", f"I can see why: your line has used {ln.get('data_used_gb')} GB of its {self.plan.get('data_limit_gb') if self.plan else '?'} GB "
                                   f"monthly data limit, and mobile data stops once the limit is reached. I can add more data to your line at "
                                   f"${price:.2f} per GB, up to 2 GB. How much data would you like to add?", "refuel_offer")
            if rf["stage"] in ("offered", "confirm"):
                m = GB.search(t)
                if m and rf["stage"] == "offered":
                    x = float(m.group(1) or WORDNUM.get((m.group(2) or "").lower(), m.group(2) or 0))
                    rf["gb"] = min(2.0, max(0.0, x))
                if rf["gb"] and (rf["stage"] == "confirm" and YES.search(t) or rf["stage"] == "offered" and "$" in t):
                    rf["stage"] = "done"; tr.usage_exceeded = False
                    self.after_tool = f"Done: I've added {rf['gb']:g} GB of data to your line for ${rf['gb'] * price:.2f}. {self.retest_now(tr)}"
                    return ("tool", ("refuel_data", {"customer_id": cid, "line_id": lid, "gb_amount": rf["gb"]}), "refuel")
                if rf["gb"] and rf["stage"] == "offered":
                    rf["stage"] = "confirm"
                    return ("message", f"To confirm: adding {rf['gb']:g} GB will cost ${rf['gb'] * price:.2f}, added to your bill. Shall I go ahead?", "refuel_confirm")
                if rf["stage"] == "offered" and not rf.get("options_sent"):
                    rf["options_sent"] = True          # v5c: the policy says ask how much; no recommendation from the model
                    return ("message", f"You can add 1 GB (${price:.2f}) or 2 GB (${2 * price:.2f}); 2 GB is the most that can be added. Which would you like?", "refuel_options")
                return None
        # ---- roaming: customer abroad and the line's roaming is off -> enable it (no cost), then the phone side
        if self.abroad and not self.roam_done and ln and ln.get("roaming_enabled") is False:
            self.roam_done = True; st.setdefault("sent", set()).add("roaming")
            tr.pending.add("roaming")
            self.after_tool = (f"I've turned on roaming for your line at no cost to you. Please also make sure Data Roaming is turned on in your "
                               f"phone's settings, {self.retest(tr)}.")
            return ("tool", ("enable_roaming", {"customer_id": cid, "line_id": lid}), "roaming_line")
        # ---- resolved
        if tr.symptom == "ok":
            if not st.get("closed"):
                st["closed"] = True; return ("message", T.CLOSING, "closing")
            return None
        if T.OPEN_QUESTION.search(t) and (T.POLICY_Q.search(t) or T.HOWTO_Q.search(t)) and not T.PROCEED_Q.search(t):
            return None
        fx = self.next_fix(tr)                       # v5f: a fault reported in any message is fixed first (at most twice)
        if fx:
            return fx
        for step in sorted(tr.unanswered):
            if step in T.REASK_LABEL and step in st.get("sent", set()) and step not in st.setdefault("reasked", set()) and step not in ("line", "usage", "reboot", "apn_reset"):
                st["reasked"].add(step); tr.pending.add(step)
                return ("message", T.REASK.format(label=T.REASK_LABEL[step]), "reask")
        while self.redo:                             # v5f: a step answered without a result is re-asked once
            step = self.redo.pop(0)
            if step in self.result_seen or step in self.redo_sent:
                continue
            self.redo_sent.add(step); tr.pending.add(step); self.last_step = step; self.last_steps = [step]
            return ("message", T.REASK.format(label=T.REASK_LABEL[step]), "redo")
        nxt = self.priority if (self.priority and self.priority not in st.get("sent", set()) and self.priority not in tr.handled) else self.next_step(tr)
        self.priority = None
        if nxt is None:
            return self.next_readout(tr)             # v5f: readouts that reveal the remaining fault, then the LLM
        st.setdefault("sent", set()).add(nxt); self.last_step = nxt; self.last_steps = [nxt]
        if nxt == "usage":
            tr.handled.add("usage")
            return ("tool", ("get_data_usage", {"customer_id": cid, "line_id": lid}), "usage")
        if BUNDLE and not self.no_bundle and nxt in T.STEP_TEMPLATE and nxt not in ("roaming", "reboot", "apn_reset", "apn_mms"):
            # v5d: bundle a second untried phone step into the same message (episodes with many faults otherwise hit the
            # 200-message limit); any step the customer skips is re-asked once by the existing rule
            second = self.next_step(tr)
            if second and second in T.STEP_TEMPLATE and second not in ("roaming", "usage", "line", "reboot", "apn_reset", "apn_mms") and second != nxt:
                st["sent"].add(second); self.last_steps = [nxt, second]
                a = T.STEP_TEMPLATE[nxt].split(", {retest}")[0].rstrip(",")
                b = T.STEP_TEMPLATE[second].split(", {retest}")[0].rstrip(",")
                return ("message", f"Please check two things: (1) {a}. (2) {b}. Then {self.retest(tr).replace('then ', '', 1)}.", f"{nxt}+{second}")
        if nxt == "roaming" and self.abroad is False:
            tr.handled.add("roaming"); return self.move(tr, last_user_text, False)
        if nxt == "roaming" and self.abroad is None and not self.asked_travel:
            st["sent"].discard("roaming")                   # ask first; the roaming move follows from the answer
            self.asked_travel = True; self.awaiting_travel = True
            return ("message", ASK_TRAVEL, "ask_travel")
        if nxt == "roaming" and self.abroad and ln.get("roaming_enabled") is not False:
            pass                                             # line roaming already on: only the phone-side instruction
        return ("message", T.STEP_TEMPLATE[nxt].format(retest=self.retest(tr)), nxt)
