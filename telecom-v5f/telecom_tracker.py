"""State tracker for tau2-telecom (agent-visible information only).

The decision state is small and listed up front (108 states):
  intent    service | data | mms          from the customer's first message (keywords)
  symptom   unk | fail | ok               last report of the intent's own test (status bar / speed test / MMS)
  untried   0 | 1 | 2 | 3+                manual steps relevant to the intent that the agent has not yet handled
  blocker   none | sim_locked | contract_ended
                                          the two situations where policy says transfer is correct (SIM locked with PIN/PUK;
                                          line suspended and its contract end date is in the past, so it may not be lifted)
Inputs used: the customer's messages (keywords), the agent's own messages (which setting it asked about) and the agent's
tool results (exact JSON). The customer's phone tool calls/results are NOT visible to the agent and are used only offline
to score the tracker.
"""
import json, re

INTENTS = ("service", "data", "mms")
SYMPTOMS = ("unk", "fail", "ok")
UNTRIED = (0, 1, 2, 3)
BLOCKERS = ("none", "sim_locked", "contract_ended")
POLICY_NOW = "2025-02-25"            # "The current time is 2025-02-25" (telecom main policy)

# manual steps: keywords in an AGENT message that mean the agent asked the customer to check/change that setting,
# and the customer phone tools that carry the step out (for offline scoring only)
STEPS = {
    "airplane":     (r"airplane", {"toggle_airplane_mode"}),
    "sim":          (r"\bsim\b", {"check_sim_status", "reseat_sim_card"}),
    "apn_reset":    (r"reset (?:your |the )?apn|apn settings? (?:to|back to) (?:the )?default|reset apn", {"reset_apn_settings"}),
    "reboot":       (r"restart|reboot", {"reboot_device"}),
    "roaming":      (r"roaming", {"toggle_roaming"}),
    "mobile_data":  (r"mobile data (?:on|off|toggle|setting|switch)|turn (?:on|off) (?:your )?mobile data|toggle (?:your )?(?:mobile )?data(?! saver)|mobile data is (?:on|off|enabled|disabled)", {"toggle_data"}),
    "data_saver":   (r"data saver|data restriction", {"toggle_data_saver_mode", "check_data_restriction_status"}),
    "netpref":      (r"network mode|network (?:type )?preference|preferred network", {"set_network_mode_preference", "check_network_mode_preference"}),
    "vpn":          (r"\bvpn\b", {"check_vpn_status", "disconnect_vpn"}),
    "wifi_calling": (r"wi-?fi calling", {"check_wifi_calling_status", "toggle_wifi_calling"}),
    "apn_mms":      (r"mmsc|apn", {"check_apn_settings", "set_apn_settings", "reset_apn_settings"}),
    "app_perm":     (r"app(?:'s)? permissions?|permissions? (?:for|of|to) (?:the |your )?(?:messaging|messages|sms|storage)|grant\b.{0,40}permission|(?:sms|storage) permission|messaging app.{0,30}permission", {"check_app_permissions", "grant_app_permission"}),
}
# what the customer's reply must mention for an asked-about step to count as handled
REPLY_KW = {"airplane": r"airplane", "sim": r"\bsim\b", "apn_reset": r"\bapn\b|reset", "reboot": r"restart|reboot|turned (?:it|my phone|the phone) (?:off|back on)",
            "roaming": r"roaming", "mobile_data": r"mobile data|data (?:is |was )?(?:now )?(?:on|off|enabled|disabled)",
            "data_saver": r"saver|restriction", "netpref": r"network mode|preference|preferred|\b[2345]g\b",
            "vpn": r"\bvpn\b", "wifi_calling": r"wi-?fi calling", "apn_mms": r"\bapn\b|mmsc", "app_perm": r"permission|\bsms\b|storage"}
AGENT_STEPS = {"line": {"get_details_by_id", "resume_line", "send_payment_request"},
               "usage": {"get_data_usage", "refuel_data"}}
SERVICE_STEPS = ["airplane", "sim", "apn_reset", "reboot", "line"]
DATA_STEPS = SERVICE_STEPS + ["roaming", "mobile_data", "usage", "data_saver", "netpref", "vpn"]
MMS_STEPS = DATA_STEPS + ["wifi_calling", "apn_mms", "app_perm"]
RELEVANT = {"service": SERVICE_STEPS, "data": DATA_STEPS, "mms": MMS_STEPS}
STEP_LABEL = {"airplane": "Airplane Mode", "sim": "SIM card status / re-seat", "apn_reset": "reset APN settings to default",
              "reboot": "restart the device", "line": "line status / suspension (agent tools)", "roaming": "data roaming (phone and line)",
              "mobile_data": "mobile data on/off", "usage": "data usage / refuel (agent tools)", "data_saver": "Data Saver",
              "netpref": "network mode preference", "vpn": "VPN", "wifi_calling": "Wi-Fi Calling", "apn_mms": "APN / MMSC settings",
              "app_perm": "messaging app permissions (SMS, storage)"}
PHONE = re.compile(r"(?<!\d)\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})(?!\d)")
DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u00a0": " "})
NEG = re.compile(r"\b(?:not|isn't|isnt|wasn't|never|no longer|no|don't|doesn't|didn't)\b")


def norm(text):
    return (text or "").translate(DASHES).lower()


def all_states():
    return [(i, s, u, b) for i in INTENTS for s in SYMPTOMS for u in UNTRIED for b in BLOCKERS]


def key_str(k):
    return "|".join(str(x) for x in k)


INTENT_RANK = {"service": 0, "data": 1, "mms": 2}


MMS_CUE = re.compile(r"\bmms\b|picture messag|multimedia messag|group (?:message|text)|(?:send|sending) (?:a |the )?(?:picture|photo|image)s?\b|picture text|photo messag")
DATA_CUE = re.compile(r"mobile data|\binternet\b|data (?:is|isn't|isnt|not|stopped|stops|keeps|connection|speed)|slow (?:data|internet|speed|connection)|speed test|browsing|web ?pages?|load(?:ing)? (?:pages|websites|apps)")
SVC_CUE = re.compile(r"no service|no signal|can't make (?:or receive )?calls|cannot make calls|calls? (?:won't|don't|aren't)|emergency|\bsos\b|no bars")


def classify_intent(text):
    """(intent, has_cue): the intent suggested by one customer message and whether it carried an explicit cue."""
    t = norm(text)
    if MMS_CUE.search(t):
        return "mms", True
    if DATA_CUE.search(t):
        return "data", True
    return "service", bool(SVC_CUE.search(t))


MMS_FAIL = re.compile(r"(?:can(?:not|'t| not)|couldn't|could not|wasn't able to|was not able to|unable to|not able to)\s+(?:send|get)"
                      r"|(?:never|didn't|did not|doesn't|does not|don't|do not|won't|isn't|is not|not)\s+(?:go(?:ing)?|went|gone)\s+through"
                      r"|(?:didn't|did not|doesn't|does not|won't|isn't|is not|still not|not)\s+(?:send|sending|working|work)\b"
                      r"|still (?:not|won't|isn't|can't|doesn't|fail)|nothing (?:is )?working|could not be sent|failed to send|\bfail(?:s|ed|ing)?\b")
MMS_OK = re.compile(r"(?:mms|picture|photo|message)[^.?!]{0,40}\b(?:sent|went through|go(?:es)? through|worked|is working|works)\b"
                    r"|\b(?:it|that) (?:worked|works)\b|sent successfully|was able to send|can now send|now i can send|can send (?:mms|picture|pictures|photos|the picture|it)(?: now)?\b")


def _last(pattern, t):
    m = None
    for m in re.finditer(pattern, t):
        pass
    return m.start() if m else -1


def symptom_from(intent, text):
    t = norm(text)
    if intent == "mms":
        if MMS_FAIL.search(t):
            return "fail"
        if MMS_OK.search(t) and not re.search(r"(?:see|check|try) (?:if|whether) i can send", t):
            return "ok"
        return None
    if intent == "data":
        if re.search(r"no connection|not working at all|speed test failed", t):
            return "fail"
        anchors = [m.start() for m in re.finditer(r"mbps", t)] or [m.start() for m in re.finditer(r"speed test|speed|result", t)]
        rates = [(m.start(), m.group(1)) for m in re.finditer(r"\b(very poor|poor|fair|good|excellent)\b", t)]
        if anchors and rates:                       # the rating word nearest to the measured value (Mbps) decides
            pos, word = min(rates, key=lambda r: min(abs(r[0] - a) for a in anchors))
            return "ok" if word == "excellent" else "fail"
        return None
    fail = _last(r"no signal|no service|emergency (?:calls )?only|airplane mode (?:is )?(?:on|icon|symbol)|airplane (?:icon|symbol)", t)
    ok = _last(r"\b(?:excellent|good|fair|poor)\b[^.?!]{0,30}\b(?:signal|bars?|[2345]g|lte)\b|\b(?:signal|bars?)\b[^.?!]{0,30}\b(?:excellent|good|fair|poor)\b|service (?:is )?(?:back|restored)|i (?:now )?have service|signal is back", t)
    if fail < 0 and ok < 0:
        return None
    return "ok" if ok > fail else "fail"


class Tracker:
    def __init__(self):
        self.intent = None
        self.intent_frozen = False
        self.symptom = "unk"
        self.handled = set()          # steps the agent asked about and the customer has answered since
        self.pending = set()          # steps asked about in the agent's last message, awaiting the customer's reply
        self.phone = None
        self.customer = None          # customer record (dict) once found
        self.lines_checked = {}       # line_id -> line record
        self.target_line = None
        self.overdue = None           # True/False once bills are seen
        self.sim_locked = False
        self.n_user = 0
        self.s1_tried = set()         # System-1 calls already issued (never repeated)
        self.unanswered = set()       # steps asked about in the agent's last message that the reply did not mention
        self.usage_exceeded = False   # get_data_usage showed the limit exceeded and no refuel yet
        self.s1_disabled = False      # set after any System-1 call errors: the LLM takes over lookups

    # ---- inputs
    def user(self, text):
        self.n_user += 1
        if not self.intent_frozen:                  # read messages until one carries an explicit cue, then freeze
            c, cue = classify_intent(text)
            self.intent = c
            self.intent_frozen = cue
        m = PHONE.search(text or "")
        if m and not self.phone:
            self.phone = "-".join(m.groups())
        t = norm(text)
        for sent in re.split(r"(?<=[.!?])[\"'\u201d\u2019)]*\s+", t):     # v5 fix: also split after a closing quote
            if re.search(r"\blocked\b", sent) and re.search(r"\bpin\b|\bpuk\b|\bsim\b", sent) and not NEG.search(sent) \
                    and not re.search(r"screen|phone is locked|lock screen", sent):
                self.sim_locked = True
        s = symptom_from(self.intent, text)
        if s:
            self.symptom = s
        self.unanswered = set()
        for step in self.pending:                   # an asked-about step counts only if the reply mentions it
            if re.search(REPLY_KW[step], t):
                self.handled.add(step)
            else:
                self.unanswered.add(step)
        self.pending = set()

    def agent_text(self, text):
        t = norm(text)
        for step, (kw, _) in STEPS.items():
            if re.search(kw, t):
                self.pending.add(step)

    def agent_tool(self, name, args, result):
        if name in AGENT_STEPS["line"]:
            self.handled.add("line")
        if name in AGENT_STEPS["usage"]:
            self.handled.add("usage")
        if name in ("enable_roaming", "disable_roaming"):
            self.handled.add("roaming")
        try:
            r = json.loads(result) if isinstance(result, str) else result
        except Exception:
            r = None
        if name.startswith("get_customer_by") and isinstance(r, dict) and r.get("customer_id"):
            self.customer = r
        elif name.startswith("get_customer_by") and isinstance(r, list) and r and isinstance(r[0], dict):
            self.customer = r[0]
        if name == "get_details_by_id" and isinstance(r, dict) and r.get("line_id"):
            self.lines_checked[r["line_id"]] = r
            if self.phone and r.get("phone_number") == self.phone:
                self.target_line = r["line_id"]
        if name == "resume_line" and isinstance(r, dict) and r.get("line"):
            self.lines_checked[r["line"]["line_id"]] = r["line"]
        if name == "get_data_usage" and isinstance(r, dict):
            try:
                self.usage_exceeded = float(r.get("data_used_gb", 0)) >= float(r.get("data_limit_gb", 1e9)) + float(r.get("data_refueling_gb", 0))
            except Exception:
                pass
        if name == "refuel_data" and isinstance(r, dict) and "new_data_refueling_gb" in r:
            self.usage_exceeded = False
        if name == "get_bills_for_customer" and isinstance(r, list):
            self.overdue = any(str(b.get("status", "")).lower() == "overdue" for b in r if isinstance(b, dict))

    # ---- state
    def blocker(self):
        if self.sim_locked:
            return "sim_locked"
        ln = self.lines_checked.get(self.target_line) if self.target_line else None
        if ln and str(ln.get("status", "")).lower() == "suspended" and str(ln.get("contract_end_date") or "9999") < POLICY_NOW:
            return "contract_ended"
        return "none"

    def untried(self):
        rel = RELEVANT.get(self.intent or "service")
        return [s for s in rel if s not in self.handled]

    def state(self):
        n = len(self.untried())
        return (self.intent or "service", self.symptom, min(n, 3), self.blocker())

    def system1_lookup(self):
        """A lookup whose arguments are fully determined by what is already known, or None. Never repeats a call."""
        if self.s1_disabled:
            return None
        cand = None
        if self.phone and not self.customer:
            cand = ("get_customer_by_phone", {"phone_number": self.phone})
        elif self.customer and self.phone and not self.target_line:
            for lid in self.customer.get("line_ids", []) or []:
                if lid not in self.lines_checked:
                    cand = ("get_details_by_id", {"id": lid}); break
        if cand is None:
            return None
        k = cand[0] + json.dumps(cand[1], sort_keys=True)
        if k in self.s1_tried:
            return None
        self.s1_tried.add(k)
        return cand


# ---- dialogue-state-tracking mode: templated next-step instructions (content written once, used without the LLM)
RETEST = {"service": "then check your status bar and tell me what it shows",
          "data": "then run a speed test and tell me the result",
          "mms": "then try sending a picture message (MMS) again and tell me whether it goes through"}
STEP_TEMPLATE = {
    "airplane": "Please check whether Airplane Mode is turned on. If it is, turn it off, {retest}.",
    "sim": "Please check your SIM card status. If the SIM shows as missing, take the SIM card out and re-insert it (re-seat it), {retest}.",
    "apn_reset": "Please reset your APN settings to the default, restart your phone, {retest}.",
    "reboot": "Please restart your phone, {retest}.",
    "roaming": "Are you currently travelling outside your home area? If you are, please make sure Data Roaming is turned on in your phone settings, {retest}.",
    "mobile_data": "Please check that Mobile Data is turned on. If it is off, turn it on, {retest}.",
    "data_saver": "Please check whether Data Saver mode is on. If it is, turn it off, {retest}.",
    "netpref": "Please check your network mode preference. If it is set to an older mode such as 2G or 3G only, change it to 4G/5G preferred, {retest}.",
    "vpn": "Please check whether a VPN is connected on your phone. If it is, disconnect it, {retest}.",
    "wifi_calling": "Please check whether Wi-Fi Calling is turned on. If it is, turn it off, {retest}.",
    "apn_mms": "Please check your APN settings, including the MMSC URL used for picture messages. If they look wrong or the MMSC URL is not set, reset the APN settings to default, restart your phone, {retest}.",
    "app_perm": "Please check the permissions of your messaging app (the app is called \"messaging\"). If SMS or storage permission is missing, grant it, {retest}.",
}
OPEN_QUESTION = re.compile(r"\?\s*$")
PROCEED_Q = re.compile(r"what (?:should|do|can) (?:i|we) (?:do|try)|what(?:'s| is) next|what else|anything else (?:i|we) (?:can|should)|next step", re.I)


def dst_next(tr, order, last_user_text, pending_unanswered):
    """The templated System-1 move for this turn, or None to defer to the LLM.
    Returns ("message", text, step) or ("tool", (name, args), step)."""
    if tr.intent is None or not tr.customer or not tr.target_line:
        return None                                        # identity and line first: the LLM handles that dialogue
    if tr.symptom == "ok" or tr.blocker() != "none" or pending_unanswered or tr.usage_exceeded:
        return None                                        # resolved / must transfer / unanswered / refuel conversation: LLM
    ln = tr.lines_checked.get(tr.target_line) or {}
    if str(ln.get("status", "")).lower() == "suspended":
        return None                                        # payment / resume flow under the policy: LLM
    t = (last_user_text or "").strip()
    if OPEN_QUESTION.search(t) and not PROCEED_Q.search(t):
        return None                                        # the customer asked something: let the LLM answer
    untried = [x for x in order.get(tr.intent, RELEVANT[tr.intent]) if x not in tr.handled]
    if not untried:
        return None
    nxt = untried[0]
    if nxt == "line":
        tr.handled.add("line"); return dst_next(tr, order, last_user_text, pending_unanswered)   # looked up already
    if nxt == "usage":
        tr.handled.add("usage")                            # never re-issued, even if the call errors
        return ("tool", ("get_data_usage", {"customer_id": tr.customer.get("customer_id"), "line_id": tr.target_line}), nxt)
    return ("message", STEP_TEMPLATE[nxt].format(retest=RETEST[tr.intent]), nxt)


# ---- wide template coverage (mode "dstw"): the same table, with more turns written from fixed text
POLICY_Q = re.compile(r"\b(?:bill|charge|cost|price|plan|pay|payment|refund|fee|why|policy|contract|money|\$)", re.I)
HOWTO_Q = re.compile(r"how (?:do|can|would) i|where (?:do|is|can)|can't find|cannot find|don't see (?:it|that|the option)|not sure how", re.I)
FIRST_REPLY = "I'm sorry you're having this problem, and I'm glad to help. Could you please tell me the phone number of the line that has the problem?"
REASK = "Could you please check {label} on your phone and tell me exactly what you see?"
CLOSING = "Great, I'm glad it's working now. Is there anything else I can help you with today?"
HOLD = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
REASK_LABEL = {k: v for k, v in STEP_LABEL.items()}


def dst_next_wide(tr, order, last_user_text, st):
    """Wide coverage. st is the agent's per-episode dict: asked_phone, reasked (set), closed, transferred.
    Returns ("message", text, tag) | ("tool", (name, args), tag) | None (defer to the LLM)."""
    t = (last_user_text or "").strip()
    if st.get("transferred") and not st.get("held"):
        st["held"] = True; return ("message", HOLD, "hold")
    if not tr.phone:
        if not st.get("asked_phone"):
            st["asked_phone"] = True; return ("message", FIRST_REPLY, "ask_phone")
        return None
    if not tr.customer or not tr.target_line:
        return None                                        # lookups run as System-1; anything odd goes to the LLM
    b = tr.blocker()
    if b != "none" and not st.get("transferred"):
        st["transferred"] = True
        return ("tool", ("transfer_to_human_agents", {"summary": f"Customer issue ({tr.intent}) cannot be resolved: {b.replace('_', ' ')}."}), "transfer")
    if tr.symptom == "ok":
        if not st.get("closed"):
            st["closed"] = True; return ("message", CLOSING, "closing")
        return None
    if tr.usage_exceeded:
        return None
    ln = tr.lines_checked.get(tr.target_line) or {}
    if str(ln.get("status", "")).lower() == "suspended":
        return None
    if OPEN_QUESTION.search(t) and (POLICY_Q.search(t) or HOWTO_Q.search(t)) and not PROCEED_Q.search(t):
        return None                                        # policy or how-to questions: the LLM answers
    for step in sorted(tr.unanswered):                      # the customer skipped a step: re-ask it once
        if step in REASK_LABEL and step in st.get("sent", set()) and step not in st.setdefault("reasked", set()) and step not in ("line", "usage"):
            st["reasked"].add(step); tr.pending.add(step)
            return ("message", REASK.format(label=REASK_LABEL[step]), "reask")
    sent = st.setdefault("sent", set())                     # each step's template is sent at most once (plus one re-ask)
    untried = [x for x in order.get(tr.intent, RELEVANT[tr.intent]) if x not in tr.handled and x not in sent]
    while untried and untried[0] == "line":
        tr.handled.add("line"); untried = untried[1:]
    if not untried:
        return None
    nxt = untried[0]
    sent.add(nxt)
    if nxt == "usage":
        tr.handled.add("usage")
        return ("tool", ("get_data_usage", {"customer_id": tr.customer.get("customer_id"), "line_id": tr.target_line}), nxt)
    return ("message", STEP_TEMPLATE[nxt].format(retest=RETEST[tr.intent]), nxt)
