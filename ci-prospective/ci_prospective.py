"""Prospective CI-triage test: freeze now, score failures that have not happened yet, with a pre-announced price change.

freeze  (run once, now): fits the DP's model on ALL labelled history in the sample (every failure before the freeze), re-tunes
        the best rule on the same history, and forecasts each policy's cost per failure from the most recent 30 days
        (push-clustered bootstrap), for two price regimes:
          week 1 (freeze .. 2026-10-14 00:00 UTC): published prices
          week 2 (2026-10-14 .. 2026-10-21 00:00 UTC): macOS runner minutes x5, Windows x2 (announced at freeze time)
        The DP re-solves for week 2 with no new data; the rule is re-tuned on the frozen history under week-2 prices; the
        informed LLM gets the week-2 price list. Writes PREREG_ci_prospective.json + frozen model, and prints its sha256.
score   (run after 2026-10-23, when Mozilla's staff have labelled the window): loads the window's failures (crawled daily),
        the LLM decisions made later with the frozen prompts, and scores every policy with the same simulator; compares
        with the frozen forecasts.
usage: python ci_prospective.py freeze <sample.jsonl> <out_dir>
       python ci_prospective.py score <out_dir> <window.jsonl> [--ctl NAME=decisions.jsonl ...]
"""
import argparse, collections, datetime as dt, hashlib, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_triage as C

WEEK2_START = dt.datetime(2026, 10, 14, tzinfo=dt.timezone.utc).timestamp()
WINDOW_END = dt.datetime(2026, 10, 21, tzinfo=dt.timezone.utc).timestamp()
PRICES_W1 = dict(C.PRICE_PER_MIN)
PRICES_W2 = dict(C.PRICE_PER_MIN, mac=C.PRICE_PER_MIN["mac"] * 5, windows=C.PRICE_PER_MIN["windows"] * 2)
CFG = dict(price_mult=1, llm_mult=1, triage_mult=1, miss_mult=1, fb_mult=1, depth=C.cfgdepth)


def rules():
    R = {}
    for k in (0, 1, 2):
        for ap in ("star", "escalate"):
            for af in ("backout", "escalate"):
                for mx in ("star", "escalate"):
                    if k == 0 and (af != "backout" or mx != "star"):
                        continue
                    R[f"k{k}:{ap}/{af}/{mx}"] = C.pol_fixed(k, ap, af, mx)
                    if k:
                        R[f"k{k}mac0:{ap}/{af}/{mx}"] = C.pol_fixed(k, ap, af, mx, k_mac=0)
    R["k0:backout"] = C.pol_fixed(0, "backout", "backout", "backout")
    return R


def costs_for(rows, pol, model, prices, llm, seed=7):
    C.PRICE_PER_MIN.clear(); C.PRICE_PER_MIN.update(prices)
    rng = random.Random(seed); out = {}
    for r in rows:
        U = [rng.random() for _ in range(C.cfgdepth + 1)]
        out[r["job"]["id"]] = C.simulate(r, pol, model, CFG, llm, U)[0]
    return out


def boot(per, rows, n=2000):
    push = collections.defaultdict(list)
    for r in rows:
        push[r["job"]["push_id"]].append(per[r["job"]["id"]])
    keys = list(push); rng = random.Random(3); B = []
    for _ in range(n):
        smp = [push[keys[rng.randrange(len(keys))]] for _ in keys]
        B.append(sum(x for v in smp for x in v) / max(1, sum(len(v) for v in smp)))
    B.sort(); return round(sum(per.values()) / max(1, len(per)), 3), (round(B[int(.025 * n)], 3), round(B[int(.975 * n)], 3))


def freeze(sample, outdir):
    os.makedirs(outdir, exist_ok=True)
    R = [json.loads(x) for x in open(sample)]
    R = [r for r in R if C.label(r["job"])]
    llm = {}
    vp = os.path.join(os.path.dirname(sample), "..", "..", "..", "bounded", "ci_llm_verdict.jsonl")
    vp = vp if os.path.exists(vp) else os.path.expanduser("~/dtd-bounded/bounded/ci_llm_verdict.jsonl")
    if os.path.exists(vp):
        for x in open(vp):
            v = json.loads(x); llm[v["job_id"]] = v["verdict"]
    model = C.Model(R, llm or None)                         # all labelled history before the freeze
    last = max(r["job"]["submit_timestamp"] for r in R)
    recent = [r for r in R if r["job"]["submit_timestamp"] >= last - 30 * 86400]
    RULES = rules(); fc = {}
    for regime, prices in (("week1_published", PRICES_W1), ("week2_mac5_win2", PRICES_W2)):
        best = min(RULES, key=lambda nm: sum(costs_for(R, RULES[nm], model, prices, llm).values()))   # re-tuned on frozen history
        per_dp = costs_for(recent, C.pol_dp, model, prices, llm); per_rule = costs_for(recent, RULES[best], model, prices, llm)
        d = {r["job"]["id"]: per_dp[r["job"]["id"]] - per_rule[r["job"]["id"]] for r in recent}
        fc[regime] = {"prices_per_min": prices, "best_rule": best, "dp": boot(per_dp, recent), "best_rule_cost": boot(per_rule, recent),
                      "dp_minus_rule": boot(d, recent), "forecast_basis": f"{len(recent)} failures in the 30 days before the freeze"}
    C.PRICE_PER_MIN.clear(); C.PRICE_PER_MIN.update(PRICES_W1)
    frozen_model = {"base_reg": model.base_reg, "prior_k": model.prior_k,
                    "reg": {"|".join(map(str, k)): v for k, v in model.reg.items()}, "tot": {"|".join(map(str, k)): v for k, v in model.tot.items()},
                    "rp": {"|".join(k): v for k, v in model.rp.items()}, "rn": {"|".join(k): v for k, v in model.rn.items()},
                    "llm_confusion": {"|".join(k): v for k, v in (model.llm or {}).items()}}
    json.dump(frozen_model, open(os.path.join(outdir, "ci_model_frozen.json"), "w"), indent=1)
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    here = os.path.dirname(os.path.abspath(__file__))
    prereg = {"written_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "freeze_after_last_history_utc": dt.datetime.fromtimestamp(last, dt.timezone.utc).isoformat(),
              "window": {"start": "the freeze (written_utc)", "week2_start_utc": "2026-10-14T00:00:00Z", "end_utc": "2026-10-21T00:00:00Z",
                         "repo": "autoland", "jobs": "tier-1 failed jobs (testfailed/busted) with a staff label in {fixed by commit, intermittent, infra, autoclassified intermittent, intermittent needs bugid} by 2026-10-23",
                         "sample": "every such job whose id ends in 0 or 1 (deterministic 20% sample), capped at 3,000 per week"},
              "price_change": "announced at the freeze: from 2026-10-14 00:00 UTC, macOS runner minutes cost 5x and Windows 2x the published price; Linux and all other costs unchanged",
              "policies": {"dp": "C.pol_dp with the frozen model; re-solved for week-2 prices with no new data",
                           "best_rule": {k: v["best_rule"] for k, v in fc.items()},
                           "llm_controllers": "Qwen3.8-27B (vLLM, no thinking) and Opus 5.5 (claude -p), prompts in ci_llm.py (control, informed); informed facts = ci_facts.json with week-2 prices for week-2 failures; run after the window with the frozen prompts"},
              "scoring": "ci_triage.simulate with the frozen model: real same-push retrigger outcomes when observed, otherwise frozen odds; costs at the regime's prices; push-clustered bootstrap",
              "predictions": {"P1": "DP cost per failure in each week falls inside its forecast interval",
                              "P2": "DP minus best rule < 0 in each week (95% interval below zero)",
                              "P3": "DP minus each LLM controller < 0 in each week",
                              "P4": "after the price change the DP's macOS retrigger share falls and its forecast still holds"},
              "forecasts": fc, "n_history": len(R),
              "sha256": {"ci_model_frozen.json": h(os.path.join(outdir, "ci_model_frozen.json")), "ci_triage.py": h(os.path.join(here, "ci_triage.py")),
                         "ci_llm.py": h(os.path.join(here, "ci_llm.py")), "ci_prospective.py": h(os.path.abspath(__file__))}}
    p = os.path.join(outdir, "PREREG_ci_prospective.json")
    json.dump(prereg, open(p, "w"), indent=1)
    print(json.dumps({k: prereg[k] for k in ("written_utc", "forecasts", "price_change")}, indent=1))
    print("PREREG sha256", h(p))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("a"); ap.add_argument("b", nargs="?")
    a = ap.parse_args()
    if a.cmd == "freeze":
        freeze(a.a, a.b)
    else:
        raise SystemExit("score: to be run after 2026-10-23 (window crawled daily by th_window.py)")
