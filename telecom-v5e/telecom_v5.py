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
FAULT_SEEN = {"netpref": r"\b[23]g\b(?! ?/)|(?:2g|3g) only|older mode", "airplane": r"airplane(?: mode)?[^.]{0,25}\b(?:is |was )?(?:on|enabled|showing)\b",
              "wifi_calling": r"wi-?fi calling[^.]{0,20}\b(?:is |was )?(?:on|enabled)\b", "data_saver": r"data saver[^.]{0,20}\b(?:is |was )?(?:on|enabled)\b",
              "vpn": r"\bvpn\b[^.]{0,25}\b(?:is |was )?(?:on|connected|enabled|active)\b", "mobile_data": r"mobile data[^.]{0,20}\b(?:is |was )?(?:off|disabled)\b",
              "sim": r"sim[^.]{0,25}\b(?:missing|not detected|no sim|not inserted)\b", "apn_mms": r"mmsc[^.]{0,30}(?:not set|missing|empty|blank)"}
FIX_DONE = re.compile(r"\b(?:turned|switched|toggled|changed|set it|disconnected|re-?seated|reinserted|re-?inserted|reset|granted|enabled it|disabled it|now (?:on|off|set|connected|disconnected))\b|\bi(?:'ve| have)? (?:turned|switched|changed|disconnected|reset)\b")
FIX_TEMPLATE = {"netpref": "Thanks. Please change your network mode preference to 4G/5G preferred, {retest}.",
                "airplane": "Thanks. Please turn Airplane Mode off, {retest}.", "wifi_calling": "Thanks. Please turn Wi-Fi Calling off, {retest}.",
                "data_saver": "Thanks. Please turn Data Saver mode off, {retest}.", "vpn": "Thanks. Please disconnect the VPN, {retest}.",
                "mobile_data": "Thanks. Please turn Mobile Data on, {retest}.", "sim": "Thanks. Please take the SIM card out and re-insert it (re-seat it), {retest}.",
                "apn_mms": "Thanks. Please reset your APN settings to default, restart your phone, {retest}."}
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
        if "app_perm" in (self.last_steps or [self.last_step]) and re.search(r"permission", t) and not self.perm_followup:
            # v5d: a permission counts as present only if the reply positively says so; "I don't see an SMS permission" = missing
            def present(x):
                pos = re.search(r"(?:has|have|with|granted|allowed|enabled|includes?)[^.]{0,40}\b" + x + r"\b|\b" + x + r"\b[^.]{0,25}(?:granted|allowed|enabled|is there|listed)", t)
                neg = re.search(r"(?:don'?t|do not|doesn'?t|does not|not|no|missing|isn'?t|without|lacks?|denied)[^.]{0,35}\b" + x + r"\b", t)
                return bool(pos) and not neg
            missing = [x for x in ("sms", "storage") if not present(x)]
            if missing and not re.search(r"\bi (?:granted|gave|added|enabled|turned on)\b|\bhas been granted\b|\bnow granted\b", t):
                self.perm_followup = missing
        if not FIX_DONE.search(t):                    # v5e: the reply reports a fault for a step we asked about, without fixing it
            asked = (self.last_steps or ([self.last_step] if self.last_step else [])) + list(self.st.get("reasked", set()))
            for step in asked + [x for x in FAULT_SEEN if x not in asked]:     # steps just asked first, then volunteered reports
                if step in FAULT_SEEN and step not in self.fixes_sent and re.search(FAULT_SEEN[step], t):
                    self.fix_pending = step; break
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
        if self.fix_pending and self.fix_pending not in self.fixes_sent:
            step = self.fix_pending; self.fix_pending = None; self.fixes_sent.add(step)
            return ("message", FIX_TEMPLATE[step].format(retest=self.retest(tr)), f"fix_{step}")
        if self.perm_followup and not self.perm_followup_sent:
            self.perm_followup_sent = True
            need = " and ".join(x.upper() if x == "sms" else x for x in self.perm_followup)
            return ("message", f"Your messaging app needs the {need} permission to send picture messages. Please grant the {need} permission to the "
                               f"messaging app, {self.retest(tr)}.", "perm_followup")
        for step in sorted(tr.unanswered):
            if step in T.REASK_LABEL and step in st.get("sent", set()) and step not in st.setdefault("reasked", set()) and step not in ("line", "usage", "reboot", "apn_reset"):
                st["reasked"].add(step); tr.pending.add(step)
                return ("message", T.REASK.format(label=T.REASK_LABEL[step]), "reask")
        nxt = self.priority if (self.priority and self.priority not in st.get("sent", set()) and self.priority not in tr.handled) else self.next_step(tr)
        self.priority = None
        if nxt is None:
            return None
        st.setdefault("sent", set()).add(nxt); self.last_step = nxt; self.last_steps = [nxt]
        if nxt == "usage":
            tr.handled.add("usage")
            return ("tool", ("get_data_usage", {"customer_id": cid, "line_id": lid}), "usage")
        if BUNDLE and nxt in T.STEP_TEMPLATE and nxt not in ("roaming", "reboot", "apn_reset", "apn_mms"):
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
