"""Discretize-then-decide agent for tau2-telecom. Wraps tau2's own LLMAgent (same prompt, same policy, same tools).

Controller (DTD_MODE):
  free     plain LLMAgent behaviour, plus state logging (the tracker runs but never acts)
  explore  System-1 lookups on; when the LLM proposes transfer_to_human_agents, block it with probability DTD_PBLOCK
           (at most DTD_MAX_BLOCKS times per episode) and ask it to continue with the untried relevant steps
  table    as explore, but block / allow is read from the fitted table (DTD_TABLE) for the current state
  rule     baseline: block every transfer unless the tracker sees a policy blocker (SIM locked, contract ended)
  dstw     as dst with wide template coverage (telecom_tracker.dst_next_wide): first reply, re-ask of skipped steps,
           blocker transfer + hold message, closing, and instructions even after non-policy customer questions.
  dstx     v5 full executor (telecom_v5.Executor): every move whose content is fixed by the state is executed (lookups,
           blocker transfer, bill payment + resume, refuel, roaming, phone-step templates, closing); step order by DTD_DECIDE
           (mdp = static DP order, belief = posterior over the task's faults, pomdp = belief + 2-step lookahead); the LLM only
           for uncovered turns, with the 'rule' transfer gate.
  dst      dialogue-state tracking: System-1 lookups, templated next-step instructions (telecom_tracker.dst_next, step order
           from DTD_ORDER json or the manual's order) right after a customer report, the LLM only for the other turns; the
           transfer gate as in 'rule'.
System-1 lookups (DTD_SYSTEM1=1, not in free mode): get_customer_by_phone once the customer's number is known, then
get_details_by_id over the customer's lines until the line with that number is found. No LLM call is made for these.
Every assistant message carries raw_data["dtd"] = {state, decision, executed_by, calls (per-call token usage)} so runs can
be scored and the table fitted from tau2's own results.json.
"""
import json, os, random, re, sys, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import telecom_tracker as T
import telecom_v5 as V5
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage, UserMessage
from tau2.utils.llm_utils import generate


def checklist_text(order_path):
    """The controller's knowledge as an instruction for the LLM: the measured check order per issue type and the policy flows."""
    order = json.load(open(order_path)) if order_path else {}
    names = {"mms": "Picture / MMS messages not working", "data": "Mobile data not working or slow", "service": "No service / no signal"}
    lines = ["", "## Troubleshooting checklist (use it)",
             "Work through the relevant checks one at a time in this order. For each, ask the customer to check that setting on "
             "their phone and tell you what they see; if it is wrong, have them fix it and retry what was failing."]
    for k in ("mms", "data", "service"):
        steps = [T.STEP_LABEL[x] for x in order.get(k, [])]
        lines.append(f"- {names[k]}: " + "; ".join(f"{i+1}. {x}" for i, x in enumerate(steps)))
    lines += ["Policy flows: if the line is suspended for an overdue bill, send the payment request and resume the line once it "
              "is paid; if the data limit is exceeded, offer a data refuel and confirm the price before adding it; ask whether the "
              "customer is travelling abroad as its own question before changing roaming; if a messaging-app permission is "
              "missing (SMS or storage), have the customer grant it. Transfer to a human only when policy requires it (for example "
              "a locked SIM or an ended contract) and only after trying all relevant checks."]
    return "\n".join(lines)


def checklist_full_text(order_path):
    """Same-knowledge control: the checklist plus every executor template, verbatim, so the LLM holds the executor's wording,
    fix instructions and policy-flow messages, not only the order."""
    rt = "then try again what was failing and tell me the result"
    lines = [checklist_text(order_path), "", "## Exact wording to use (the same messages a scripted agent uses)",
             "Opening: " + T.FIRST_REPLY,
             "Ask about travel before touching roaming: " + V5.ASK_TRAVEL]
    for k, v in T.STEP_TEMPLATE.items():
        lines.append(f"Check {T.STEP_LABEL[k]}: " + v.format(retest=rt))
    for k, v in V5.FIX_TEMPLATE.items():
        lines.append(f"If {T.STEP_LABEL[k]} is wrong: " + v.format(retest=rt))
    lines += ["If the messaging app lacks a permission: Your messaging app needs the <SMS / storage> permission to send picture "
              "messages. Please grant the <permission> permission to the messaging app in your phone settings, " + rt + ".",
              "If a step was skipped: " + T.REASK.format(label="<the setting>"),
              "Overdue bill: I found the cause: your line is suspended because of an overdue bill (bill <id>, <amount>). I can send "
              "you a payment request; once it is paid I will resume your line. Shall I send it?",
              "Payment not seen yet: I don't see the payment yet. Please check your payment requests and accept the request for the bill.",
              "Data limit exceeded: I can see why: your line has used <used> GB of its <limit> GB limit. I can add data (a refuel). "
              "You can add 1 GB ($<price>) or 2 GB ($<2x price>); 2 GB is the most that can be added. Which would you like?",
              "Before refuelling: To confirm: adding <n> GB will cost $<cost>, added to your bill. Shall I go ahead?",
              "When fixed: " + T.CLOSING,
              "After a transfer tool call: " + T.HOLD]
    return "\n".join(lines)


NO_LLM_MSG = "Could you please tell me what you see on your phone right now, so I can help you with the next step?"

BLOCK_NOTE = ("\n\n[controller: Do not transfer to a human agent yet, and do not tell the customer you will transfer them. "
              "The policy requires trying all relevant resolution steps first. You cannot change the phone yourself, but you can "
              "ask the customer to check or change a setting on their phone and tell you the result. Relevant steps not yet "
              "tried: {steps}. Continue with the next of these steps (one action only).]")
VERBAL_TRANSFER = re.compile(r"transfer(?:ring)? you (?:to|over to) (?:a |our )?(?:human|live|specialist|technical)|you are being transferred|"
                             r"connect you (?:with|to) (?:a |our )?(?:human|live|specialist)", re.I)


def proposes_transfer(msg):
    names = [t.name for t in (msg.tool_calls or [])]
    return "transfer_to_human_agents" in names or (not names and bool(VERBAL_TRANSFER.search(msg.content or "")))


class DtDAgent(LLMAgent):
    def __init__(self, tools, domain_policy, llm, llm_args=None):
        if os.environ.get("DTD_CHECKLIST") == "1":            # ablation: the LLM is given the controller's checklist
            domain_policy = domain_policy + "\n" + checklist_text(os.environ.get("DTD_ORDER"))
        elif os.environ.get("DTD_CHECKLIST") == "2":          # ablation: checklist plus every executor template verbatim
            domain_policy = domain_policy + "\n" + checklist_full_text(os.environ.get("DTD_ORDER"))
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.no_llm = os.environ.get("DTD_NO_LLM") == "1"      # ablation: controller only, a fixed re-ask instead of every LLM turn
        self.mode = os.environ.get("DTD_MODE", "free")
        self.p_block = float(os.environ.get("DTD_PBLOCK", "0.5"))
        self.max_blocks = int(os.environ.get("DTD_MAX_BLOCKS", "3"))
        self.use_system1 = os.environ.get("DTD_SYSTEM1", "1") == "1" and self.mode != "free"
        self.table = json.load(open(os.environ["DTD_TABLE"]))["table"] if os.environ.get("DTD_TABLE") else None
        self.order = json.load(open(os.environ["DTD_ORDER"])) if os.environ.get("DTD_ORDER") else {}
        self.decide = os.environ.get("DTD_DECIDE", "mdp")
        self.prior = V5.FaultPrior(os.environ["DTD_TASKS"], os.environ["DTD_SPLITS"]) if (self.mode == "dstx" and self.decide != "mdp") else None
        if self.mode == "table" and self.table is None:
            raise ValueError("DTD_MODE=table needs DTD_TABLE")
        self._reset()

    def _reset(self):
        self.rng = random.Random(uuid.uuid4().int)
        self.tr = T.Tracker()
        self.wide = {}
        order = self.order
        if os.environ.get("DTD_ORDER_RANDOM") == "1":         # ablation: a fresh random check order per episode
            order = {k: self.rng.sample(v, len(v)) for k, v in self.order.items()}
        self.ex = V5.Executor(order, self.decide, self.prior) if self.mode == "dstx" else None
        self.blocks = 0
        self.transferred = False      # a transfer tool call was allowed; the follow-up hold message is not gated
        self.calls_by_id = {}
        self.rng = random.Random(uuid.uuid4().int)

    def get_init_state(self, message_history=None):
        self._reset()
        return super().get_init_state(message_history)

    # ---- tracker inputs
    def _ingest(self, message):
        if isinstance(message, UserMessage):
            self.tr.user(message.content or "")
            if self.ex:
                self.ex.user(self.tr, message.content or "")
        elif isinstance(message, MultiToolMessage):
            for tm in message.tool_messages:
                self._ingest_tool(tm)
        elif isinstance(message, ToolMessage):
            self._ingest_tool(message)

    def _ingest_tool(self, tm):
        name, args = self.calls_by_id.get(tm.id, (None, {}))
        if tm.id.startswith("dtd-") and (tm.error or (tm.content or "").startswith("Error")):
            self.tr.s1_disabled = True               # a System-1 lookup failed: the LLM takes over lookups for this episode
        if name and not tm.error:
            self.tr.agent_tool(name, args, tm.content)
            if self.ex:
                self.ex.tool(self.tr, name, args, tm.content)

    def _llm(self, state, note=None):
        msgs = state.system_messages + list(state.messages)
        if note:
            last = msgs[-1]
            msgs = msgs[:-1] + [last.model_copy(update={"content": (last.content or "") + note})]
        out = generate(model=self.llm, tools=self.tools, messages=msgs, call_name="agent_response", **self.llm_args)
        u = out.usage or {}
        return out, {"prompt": u.get("prompt_tokens", 0), "completion": u.get("completion_tokens", 0)}

    def _decide_block(self, key):
        if self.blocks >= self.max_blocks:
            return False, "cap"
        if self.mode == "explore":
            return self.rng.random() < self.p_block, "random"
        if self.mode in ("rule", "dst", "dstw", "dstx"):     # the manual as a rule: never transfer unless a policy blocker is present
            return key[3] == "none", "rule"
        if self.mode == "table" and self.table is not None:
            row = self.table.get(T.key_str(key))
            return (row is not None and row.get("action") == "block"), ("table" if row else "unseen")
        return False, "free"

    def _generate_next_message(self, message, state):
        self._ingest(message)
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        key = self.tr.state()
        info = {"state": T.key_str(key), "mode": self.mode, "untried": self.tr.untried(), "decision": None,
                "executed_by": "llm", "calls": []}
        det = self.tr.system1_lookup() if self.use_system1 else None
        mv = None
        if not det and self.mode == "dst" and isinstance(message, UserMessage):
            mv = T.dst_next(self.tr, self.order, message.content, bool(self.tr.unanswered))
            if mv and mv[0] == "tool":
                det = mv[1]
        if not det and self.mode == "dstw" and (isinstance(message, UserMessage) or (self.wide.get("transferred") and not self.wide.get("held"))):
            mv = T.dst_next_wide(self.tr, self.order, message.content if isinstance(message, UserMessage) else "", self.wide)
            if mv and mv[0] == "tool":
                det = mv[1]
        if not det and self.mode == "dstx":
            is_tool = isinstance(message, (ToolMessage, MultiToolMessage))
            mv = self.ex.move(self.tr, message.content if isinstance(message, UserMessage) else "", is_tool)
            if mv and mv[0] == "tool":
                det = mv[1]
        if mv and mv[0] == "message":
            out = AssistantMessage.text(content=mv[1], usage={"prompt_tokens": 0, "completion_tokens": 0})
            info["executed_by"] = "template"; info["step"] = mv[2]
        elif det:
            if mv and mv[0] == "tool":
                info["step"] = mv[2]
            tc = ToolCall(id=f"dtd-{uuid.uuid4().hex[:16]}", name=det[0], arguments=det[1], requestor="assistant")
            out = AssistantMessage.text(content="", tool_calls=[tc], usage={"prompt_tokens": 0, "completion_tokens": 0})
            info["executed_by"] = "system1"
        elif self.no_llm:
            out = AssistantMessage.text(content=NO_LLM_MSG, usage={"prompt_tokens": 0, "completion_tokens": 0})
            info["executed_by"] = "canned"
        else:
            out, u = self._llm(state)
            info["calls"].append(u)
            if proposes_transfer(out) and not self.transferred and self.mode in ("explore", "table", "rule", "dst", "dstw", "dstx"):
                block, why = self._decide_block(key)
                info["decision"] = {"proposed": "transfer", "block": block, "why": why, "blocks_before": self.blocks}
                if block:
                    self.blocks += 1
                    steps = ", ".join(T.STEP_LABEL[s] for s in self.tr.untried()) or "re-check the original symptom"
                    note = BLOCK_NOTE.format(steps=steps)
                    info["decision"]["note"] = note
                    info["decision"]["discarded"] = {"content": (out.content or "")[:500], "tools": [t.name for t in (out.tool_calls or [])]}
                    out2, u2 = self._llm(state, note)
                    info["calls"].append(u2)
                    info["decision"]["complied"] = not proposes_transfer(out2)
                    out = out2
            tot = {"prompt_tokens": sum(c["prompt"] for c in info["calls"]),
                   "completion_tokens": sum(c["completion"] for c in info["calls"])}
            out = out.model_copy(update={"usage": tot})
        for t in out.tool_calls or []:
            self.calls_by_id[t.id] = (t.name, t.arguments)
            if t.name == "transfer_to_human_agents":
                self.transferred = True
        if out.content:
            self.tr.agent_text(out.content)
        raw = dict(out.raw_data or {})
        raw["dtd"] = info
        return out.model_copy(update={"raw_data": raw})


def create_dtd_agent(tools, domain_policy, **kwargs):
    return DtDAgent(tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"))
