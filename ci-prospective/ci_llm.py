"""Qwen on CI failures (runs against the vLLM server). Three prompts per failure, same failure text in each:
  verdict    classify: intermittent | regression                           (the LLM as a reader; its confusion feeds the DP)
  control    choose the move: star | retrigger | escalate | backout        (the LLM as the controller, no numbers)
  informed   same choice, with the price list and the measured odds in the prompt (train window only)
The failure text is what a sheriff sees in Treeherder: job name, platform, the error lines the system extracted, and the
top candidate bugs (title, occurrence count). No label, no retrigger results.
For control / informed, three questions per failure: the first move, the move after one retrigger PASSED, and after one
retrigger FAILED (so the LLM's own multi-step policy can be replayed without extra calls).
usage: python ci_llm.py <sample.jsonl> <out.jsonl> <mode: verdict|control|informed> [informed_facts.json]
"""
import concurrent.futures as cf, json, os, re, sys, urllib.request

URL = os.environ.get("LLM_URL", "http://localhost:8000/v1/chat/completions")


def failure_text(r):
    j = r["job"]
    errs = "\n".join("  " + e["line"][:300] for e in r["errors"][:12]) or "  (none extracted)"
    bugs = []
    for s in r["suggestions"][:3]:
        for b in (s["open_recent"] + s["all_others"])[:2]:
            bugs.append(f"  - bug: {b.get('summary','')[:160]} (occurrences: {b.get('occurrences')})")
    mins = (j["end_timestamp"] - j["start_timestamp"]) / 60 if j.get("end_timestamp") and j.get("start_timestamp") else None
    return (f"Job: {j['job_type_name']}\nPlatform: {j['platform']} ({j.get('platform_option')})\nResult: {j['result']}\n"
            f"Job length: {mins:.0f} minutes\n" if mins else f"Job: {j['job_type_name']}\nPlatform: {j['platform']}\nResult: {j['result']}\n") + \
           f"Error lines:\n{errs}\nCandidate bugs matched by the failure lines:\n" + ("\n".join(bugs) or "  (none)")


PROMPTS = {
    "verdict": ("You triage Firefox CI failures on the autoland branch. A failed test job is either an INTERMITTENT failure "
                "(flaky test or infrastructure; it would pass on a retrigger) or a REGRESSION caused by the pushed change.\n\n"
                "{failure}\n\nAnswer with exactly one word: intermittent or regression."),
    "control": ("You triage Firefox CI failures on the autoland branch. Choose the next move for this failed job:\n"
                "  star      - mark it as a known intermittent and move on\n"
                "  retrigger - run the same job again and look at the result\n"
                "  escalate  - ask a developer to investigate\n"
                "  backout   - back out the push that caused it\n\n{failure}\n\n"
                "Answer with exactly one word: star, retrigger, escalate or backout."),
    "informed": ("You triage Firefox CI failures on the autoland branch. Choose the next move for this failed job so as to "
                 "minimise the expected total cost.\n  star      - mark it as a known intermittent and move on\n"
                 "  retrigger - run the same job again and look at the result\n  escalate  - ask a developer to investigate\n"
                 "  backout   - back out the push that caused it\n\nCosts and measured odds:\n{facts}\n\n{failure}\n\n"
                 "Answer with exactly one word: star, retrigger, escalate or backout."),
}
WORDS = {"verdict": ("intermittent", "regression"), "control": ("star", "retrigger", "escalate", "backout"),
         "informed": ("star", "retrigger", "escalate", "backout")}


def ask(prompt):
    body = {"model": "qwen", "messages": [{"role": "user", "content": prompt}], "max_tokens": 8, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"] or "", d.get("usage", {})


def facts_text(f, r):
    j = r["job"]
    plat = j["platform"] or ""
    p = next((v for k, v in f["price_per_min"].items() if k in plat), f["price_per_min"]["linux"])
    mins = (j["end_timestamp"] - j["start_timestamp"]) / 60 if j.get("end_timestamp") and j.get("start_timestamp") else 25
    return (f"  - a retrigger of this job costs ${p*mins:.2f} (machine time)\n  - escalating costs ${f['escalate']:.2f} of developer time\n"
            f"  - starring a real regression costs ${f['miss']:.2f} later (bisect and backout)\n"
            f"  - backing out a push for an intermittent costs ${f['false_backout']:.2f} (re-landing)\n"
            f"  - {100*f['base_reg']:.0f}% of failed jobs like this are real regressions\n"
            f"  - a retrigger passes {100*f['pass_int']:.0f}% of the time for intermittent failures and {100*f['pass_reg']:.0f}% for regressions")


if __name__ == "__main__":
    src, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    F = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
    R = [json.loads(x) for x in open(src)]
    done = set()
    if os.path.exists(out):
        done = {json.loads(x)["job_id"] for x in open(out)}
    todo = [r for r in R if r["job"]["id"] not in done]

    AFTER = {"first": "", "pass": "\n\nUpdate: you already retriggered this job once and the retrigger PASSED.",
             "fail": "\n\nUpdate: you already retriggered this job once and the retrigger FAILED the same way."}

    def run(r):
        rec = {"job_id": r["job"]["id"]}
        for ctx in (["first"] if mode == "verdict" else ["first", "pass", "fail"]):
            p = PROMPTS[mode].format(failure=failure_text(r) + AFTER[ctx], facts=facts_text(F, r) if F else "")
            txt, u = ask(p)
            first = re.findall(r"[a-z]+", txt.lower())[:1]
            w = first[0] if first and first[0] in WORDS[mode] else next((w for w in WORDS[mode] if re.search(rf"\b{w}\b", txt.lower())), None)
            rec["verdict" if ctx == "first" else f"after_{ctx}"] = w
            rec.setdefault("usage", []).append(u)
        return rec

    with open(out, "a") as f, cf.ThreadPoolExecutor(32) as ex:
        for res in ex.map(run, todo):
            f.write(json.dumps(res) + "\n")
    print(f"{mode}: wrote {len(todo)}")
