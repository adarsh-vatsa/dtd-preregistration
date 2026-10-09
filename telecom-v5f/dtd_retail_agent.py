"""DtD agent for tau2-retail. Wraps tau2's LLMAgent (same prompt, policy and tools); the controller executes every move whose
content is fixed by the state and checks every database write before it happens.

State (system-defined, from tool results and regexes over the customer's text):
  identity   email / first+last name / zip seen; user_id found (find_user_*); user record (orders, payment methods)
  orders     order records fetched (status, items with product ids, payment history); product records (variants)
  write      the write the LLM proposed, its validation result, and whether the customer confirmed THAT write
Executed moves (no LLM call):
  lookups    find_user_id_by_email / _by_name_zip once the identity is known; get_user_details; get_order_details for every
             order id the customer names; get_product_details for the products in an order before a write that needs them
  validate   status (pending for cancel/modify, delivered for return/exchange), item ownership, same product and available
             variant, distinct new item, payment method owned (and gift-card balance for price differences), cancel reason;
             an invalid write is not sent: the LLM is re-asked with the violation (at most twice)
  confirm    a valid write is turned into a templated confirmation that lists exactly what will happen (order, items by name and
             options, new options, payment method, price difference); on an explicit yes the stored write is executed as is
Decide step for the confirmation (DTD_DECIDE):
  mdp     top guess: if the customer's last message is a yes, trust that it confirms the LLM's write and execute it
  qmdp    execute without a new confirmation only if the LLM's own previous message already named every key detail of the write
          (order id and every item / payment id involved); otherwise confirm
  pomdp   always buy the information: every write goes through the templated confirmation once (the customer's reply is the
          observation); default
"""
import json, os, re, sys, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage, UserMessage
from tau2.utils.llm_utils import generate

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
ZIP = re.compile(r"\b(\d{5})\b")
NAME = re.compile(r"(?:my name is|i am|i'm|this is|name'?s)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)")
ORDER = re.compile(r"#?W\d{7}")
YES = re.compile(r"^\W*(?:yes|yeah|yep|correct|confirm(?:ed)?|that'?s (?:right|correct)|go ahead|please (?:do|proceed)|proceed|sure|ok(?:ay)?)\b", re.I)
WRITES = {"cancel_pending_order", "modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment",
          "modify_user_address", "return_delivered_order_items", "exchange_delivered_order_items"}
NOTE = ("\n\n[controller: your proposed {tool} call was not sent because it would violate the policy or the data: {why}. "
        "Fix it (look up what you need first) or ask the customer.]")


class RetailState:
    def __init__(self):
        self.addr_requested = False; self.pay_requested = False; self.addr_done = set(); self.pay_done = set()
        self.email = None; self.first = self.last = None; self.zip = None
        self.user_id = None; self.user = None; self.orders = {}; self.products = {}
        self.asked = set()

    def user_text(self, text):
        t = text or ""
        if re.search(r"\baddress\b", t, re.I):
            self.addr_requested = True
        if re.search(r"payment method|pay (?:with|using)|switch (?:the )?payment|change (?:the )?payment", t, re.I):
            self.pay_requested = True
        if not self.email and (m := EMAIL.search(t)):
            self.email = m.group(0).rstrip(".")
        if not self.first and (m := NAME.search(t)):
            self.first, self.last = m.group(1), m.group(2)
        if not self.zip and (m := ZIP.search(t)):
            self.zip = m.group(1)
        return ["#" + o.lstrip("#") for o in ORDER.findall(t)]

    def tool(self, name, args, content):
        try:
            r = json.loads(content) if isinstance(content, str) else content
        except Exception:
            r = content
        if name.startswith("find_user_id") and isinstance(r, str) and r and not r.startswith("Error"):
            self.user_id = r.strip().strip('"')
        elif name == "get_user_details" and isinstance(r, dict):
            self.user = r
        if name == "modify_pending_order_address" and isinstance(args, dict):
            self.addr_done.add(args.get("order_id"))
        if name == "modify_pending_order_payment" and isinstance(args, dict):
            self.pay_done.add(args.get("order_id"))
        if name in ("get_order_details",) or name in WRITES and isinstance(r, dict) and r.get("order_id"):
            if isinstance(r, dict) and r.get("order_id"):
                self.orders[r["order_id"]] = r
        elif name == "get_product_details" and isinstance(r, dict) and r.get("product_id"):
            self.products[r["product_id"]] = r

    def pm(self, pid):
        for k, v in ((self.user or {}).get("payment_methods") or {}).items():
            if k == pid or (isinstance(v, dict) and v.get("id") == pid):
                return v
        return None


def describe_item(st, order, iid):
    for it in order.get("items", []):
        if it.get("item_id") == iid:
            return f"{it.get('name')} ({', '.join(f'{k}: {v}' for k, v in (it.get('options') or {}).items())})", it
    return iid, None


def variant(st, product_id, iid):
    p = st.products.get(product_id) or {}
    v = (p.get("variants") or {}).get(iid)
    return v


def validate(st, name, a):
    """Returns (ok, why, needed_lookups, summary)."""
    need = []
    if not st.user_id or not st.user:
        return False, "the customer is not authenticated yet (find the user id via email or name + zip, then get the user details)", need, None
    oid = a.get("order_id")
    if name != "modify_user_address":
        if not oid:
            return False, "no order id", need, None
        o = st.orders.get(oid)
        if o is None:
            return False, f"order {oid} has not been looked up", [("get_order_details", {"order_id": oid})], None
        if o.get("user_id") and o["user_id"] != st.user_id:
            return False, f"order {oid} belongs to another user", need, None
        status = str(o.get("status", ""))
        if name in ("cancel_pending_order", "modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment") and status != "pending":
            return False, f"order {oid} is '{status}', only pending orders can be cancelled or modified", need, None
        if name in ("return_delivered_order_items", "exchange_delivered_order_items") and status != "delivered":
            return False, f"order {oid} is '{status}', only delivered orders can be returned or exchanged", need, None
    if name == "cancel_pending_order":
        if a.get("reason") not in ("no longer needed", "ordered by mistake"):
            return False, "the reason must be 'no longer needed' or 'ordered by mistake'", need, None
        return True, "", need, f"cancel order {oid} (reason: {a['reason']}); the total is refunded to the original payment method"
    if name == "modify_pending_order_items" and ((st.addr_requested and oid not in st.addr_done) or (st.pay_requested and oid not in st.pay_done)):
        # policy: modifying items locks the order ("The agent will not be able to modify or cancel the order anymore"), so a
        # requested address or payment change on the same order must be made first
        if not getattr(st, "order_warned", False):
            st.order_warned = True
            return False, ("modifying items locks the order, so if the customer also wants to change this order's shipping address or payment "
                           "method, do that first (confirm it with the customer), then modify the items"), need, None
    if name in ("modify_pending_order_items", "exchange_delivered_order_items", "return_delivered_order_items"):
        o = st.orders[oid]; own = [it.get("item_id") for it in o.get("items", [])]
        items = list(a.get("item_ids") or [])
        for iid in items:
            if items.count(iid) > own.count(iid):
                return False, f"item {iid} is not in order {oid}", need, None
        pm = st.pm(a.get("payment_method_id"))
        if pm is None:
            return False, f"payment method {a.get('payment_method_id')} is not one of the customer's payment methods", need, None
        if name == "return_delivered_order_items":
            names = [describe_item(st, o, i)[0] for i in items]
            return True, "", need, f"return from order {oid}: {'; '.join(names)}; refund to {a.get('payment_method_id')}"
        new = list(a.get("new_item_ids") or [])
        if len(new) != len(items):
            return False, "item_ids and new_item_ids must have the same length (one new item per old item)", need, None
        lines = []; diff = 0.0
        for iid, nid in zip(items, new):
            if iid == nid:
                return False, f"new item {nid} is the same as the old item", need, None
            desc, it = describe_item(st, o, iid)
            pid = (it or {}).get("product_id")
            if pid and pid not in st.products:
                need.append(("get_product_details", {"product_id": pid}))
                continue
            v = variant(st, pid, nid)
            if v is None:
                return False, f"new item {nid} is not a variant of the same product as {iid} ({desc})", need, None
            if not v.get("available", True):
                return False, f"new item {nid} is not available", need, None
            diff += float(v.get("price", 0)) - float((it or {}).get("price", 0))
            lines.append(f"{desc} -> {', '.join(f'{k}: {val}' for k, val in (v.get('options') or {}).items())}")
        if need:
            return False, "product details needed", need, None
        if pm.get("source") == "gift_card" and float(pm.get("balance", 0)) < diff:
            return False, f"gift card balance {pm.get('balance')} does not cover the price difference {diff:.2f}", need, None
        verb = "modify" if name.startswith("modify") else "exchange"
        sign = f"you pay ${diff:.2f}" if diff > 0 else f"you receive a refund of ${-diff:.2f}"
        return True, "", need, (f"{verb} items in order {oid}: " + "; ".join(lines) + f"; price difference: {sign}, using {a.get('payment_method_id')}"
                                + (". This can be done only once for this order, so please make sure these are all the items you want to change" if verb == "modify" else ""))
    if name == "modify_pending_order_payment":
        pm = st.pm(a.get("payment_method_id"))
        if pm is None:
            return False, f"payment method {a.get('payment_method_id')} is not one of the customer's payment methods", need, None
        return True, "", need, f"change the payment method of order {oid} to {a.get('payment_method_id')}"
    if name in ("modify_pending_order_address", "modify_user_address"):
        addr = ", ".join(str(a.get(k)) for k in ("address1", "address2", "city", "state", "country", "zip") if a.get(k))
        tgt = f"order {oid}" if name == "modify_pending_order_address" else "your default address"
        return True, "", need, f"change the shipping address of {tgt} to {addr}"
    return True, "", need, f"{name} {json.dumps(a)}"


def mentions_all(text, name, a):
    t = text or ""
    keys = [a.get("order_id")] + list(a.get("item_ids") or []) + list(a.get("new_item_ids") or []) + [a.get("payment_method_id")]
    return all(k is None or str(k).lstrip("#") in t for k in keys)


class DtDRetailAgent(LLMAgent):
    def __init__(self, tools, domain_policy, llm, llm_args=None):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.mode = os.environ.get("DTD_MODE", "free")
        self.decide = os.environ.get("DTD_DECIDE", "pomdp")
        self._reset()

    def _reset(self):
        self.st = RetailState(); self.calls = {}; self.pending_write = None; self.queue = []; self.last_agent_text = ""

    def get_init_state(self, message_history=None):
        self._reset()
        return super().get_init_state(message_history)

    def _ingest(self, message):
        if isinstance(message, UserMessage):
            for oid in self.st.user_text(message.content or ""):
                if oid not in self.st.orders and ("get_order_details", oid) not in self.st.asked:
                    self.queue.append(("get_order_details", {"order_id": oid}))
        tms = message.tool_messages if isinstance(message, MultiToolMessage) else ([message] if isinstance(message, ToolMessage) else [])
        for tm in tms:
            name, args = self.calls.get(tm.id, (None, {}))
            if name and not tm.error:
                self.st.tool(name, args, tm.content)
                if name == "get_user_details" and isinstance(self.st.user, dict):
                    pass

    def _system1(self):
        st = self.st
        if not st.user_id:
            if st.email and ("find_email", st.email) not in st.asked:
                st.asked.add(("find_email", st.email)); return ("find_user_id_by_email", {"email": st.email})
            if st.first and st.zip and ("find_name", st.first, st.zip) not in st.asked:
                st.asked.add(("find_name", st.first, st.zip))
                return ("find_user_id_by_name_zip", {"first_name": st.first, "last_name": st.last, "zip": st.zip})
            return None
        if st.user is None and ("user", st.user_id) not in st.asked:
            st.asked.add(("user", st.user_id)); return ("get_user_details", {"user_id": st.user_id})
        while self.queue:
            name, args = self.queue.pop(0)
            k = (name, json.dumps(args, sort_keys=True))
            if k in st.asked:
                continue
            st.asked.add(k)
            if name == "get_order_details" and args["order_id"] in st.orders:
                continue
            if name == "get_product_details" and args["product_id"] in st.products:
                continue
            return (name, args)
        return None

    def _llm(self, state, note=None):
        msgs = state.system_messages + list(state.messages)
        if note:
            last = msgs[-1]
            msgs = msgs[:-1] + [last.model_copy(update={"content": (last.content or "") + note})]
        out = generate(model=self.llm, tools=self.tools, messages=msgs, call_name="agent_response", **self.llm_args)
        u = out.usage or {}
        return out, {"prompt": u.get("prompt_tokens", 0), "completion": u.get("completion_tokens", 0)}

    def _tool_msg(self, name, args, info, by):
        tc = ToolCall(id=f"dtd-{uuid.uuid4().hex[:16]}", name=name, arguments=args, requestor="assistant")
        info["executed_by"] = by
        return AssistantMessage.text(content="", tool_calls=[tc], usage={"prompt_tokens": 0, "completion_tokens": 0})

    def _generate_next_message(self, message, state):
        self._ingest(message)
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        info = {"mode": self.mode, "decide": self.decide, "executed_by": "llm", "calls": [], "write": None}
        out = None
        if self.mode != "free":
            # 1) a confirmed stored write is executed as is
            if self.pending_write and isinstance(message, UserMessage):
                pw, self.pending_write = self.pending_write, None
                txt = message.content or ""
                if YES.search(txt) and not re.search(r"\b(?:but|however|actually|instead|wait|change|not quite|hold on)\b", txt, re.I):
                    out = self._tool_msg(pw[0], pw[1], info, "confirmed_write"); info["write"] = {"tool": pw[0], "confirmed": True}
            # 2) deterministic lookups
            if out is None:
                det = self._system1()
                if det:
                    out = self._tool_msg(det[0], det[1], info, "system1")
        if out is None:
            out, u = self._llm(state); info["calls"].append(u)
            if self.mode != "free":
                for attempt in range(3):
                    tcs = [t for t in (out.tool_calls or []) if t.name in WRITES]
                    if not tcs:
                        break
                    tc = tcs[0]; ok, why, need, summary = validate(self.st, tc.name, dict(tc.arguments or {}))
                    if not ok and need:
                        self.queue.extend(need)
                        det = self._system1()
                        if det:
                            out = self._tool_msg(det[0], det[1], info, "system1_for_write"); info["write"] = {"tool": tc.name, "lookup_first": True}
                            break
                    if not ok:
                        info.setdefault("violations", []).append(why)
                        if attempt == 2:
                            out = AssistantMessage.text(content="I need to double-check a detail before I can do that. " + why[0].upper() + why[1:] + ". Could you confirm the details?",
                                                        usage={"prompt_tokens": 0, "completion_tokens": 0}); info["executed_by"] = "template"
                            break
                        out, u = self._llm(state, NOTE.format(tool=tc.name, why=why)); info["calls"].append(u)
                        continue
                    last_user = next((m.content for m in reversed(state.messages) if isinstance(m, UserMessage)), "") or ""
                    if self.decide == "mdp" and YES.search(last_user):
                        info["write"] = {"tool": tc.name, "confirmed": "trusted (mdp)"}; break
                    if self.decide == "qmdp" and YES.search(last_user) and mentions_all(self.last_agent_text, tc.name, dict(tc.arguments or {})):
                        info["write"] = {"tool": tc.name, "confirmed": "details already listed (qmdp)"}; break
                    self.pending_write = (tc.name, dict(tc.arguments or {}))
                    out = AssistantMessage.text(content=f"Before I go ahead, please confirm: I will {summary}. Shall I proceed? (yes/no)",
                                                usage={"prompt_tokens": 0, "completion_tokens": 0})
                    info["executed_by"] = "template"; info["write"] = {"tool": tc.name, "confirm_template": True}
                    break
            tot = {"prompt_tokens": sum(c["prompt"] for c in info["calls"]), "completion_tokens": sum(c["completion"] for c in info["calls"])}
            out = out.model_copy(update={"usage": tot})
        for t in out.tool_calls or []:
            self.calls[t.id] = (t.name, dict(t.arguments or {}))
        if out.content:
            self.last_agent_text = out.content
        raw = dict(out.raw_data or {}); raw["dtd"] = info
        return out.model_copy(update={"raw_data": raw})


def create_dtd_retail_agent(tools, domain_policy, **kwargs):
    return DtDRetailAgent(tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"))
