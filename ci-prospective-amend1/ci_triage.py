"""CI failure triage on Mozilla Treeherder: measured odds, published prices, finite-horizon DP vs rules (retrospective).

Decision unit: one failed tier-1 job (label known later from Mozilla's staff: intermittent / fixed by commit / infra).
State (system-defined, known at decision time):
  plat      linux | windows | mac | android | other        (price of a retrigger depends on it)
  dur       short (<15 min) | mid (15-40) | long (>40)     (job length; also prices a retrigger)
  suite     mochitest | wpt | xpcshell | reftest | build | other
  newline   any failure line flagged new in this revision (system bug-suggestion field)
  known     the top failure line matches an open bug whose title starts with "Intermittent"
  then belief after observations: retrigger results (pass/fail), LLM verdict (intermittent/regression)
Actions:
  star       accept as intermittent and move on         cost 0; if it was a real regression: C_MISS
  backout    back out the push                          cost 0 if regression; if intermittent: C_FALSE_BACKOUT
  retrigger  run the job again                          cost minutes x price(plat); observe pass/fail
  llm        Qwen reads the error lines                 cost tokens x price; observe its verdict (confusion measured)
  escalate   a person investigates                      cost TRIAGE_MIN x wage; reveals the truth, then the right terminal action
Odds: P(regression | state) from the train window; P(retrigger passes | label, suite, plat) from observed same-push runs;
LLM confusion from its verdicts on train-window jobs. Costs: published prices (defaults below; every one is a flag).
Evaluation: expected cost per failure on the TEST window (later dates), using the true label and the measured outcome
model; policies: DP, rules, LLM-as-controller (and informed LLM when its decisions are supplied).
usage: python ci_triage.py <sample.jsonl> [--llm verdicts.jsonl] [--split-date 2026-09-06]
"""
import argparse, collections, itertools, json, math, sys

PRICE_PER_MIN = {"linux": 0.006, "windows": 0.010, "mac": 0.062, "android": 0.010, "other": 0.006}   # GitHub-hosted runner list prices
WAGE_PER_HOUR = 64.44            # BLS median software developer, May 2025
TRIAGE_MIN = 15.0                # minutes a person spends investigating one failure (assumption; swept)
C_MISS_HOURS = 2.0               # developer hours lost when a real regression is starred (later bisect + backout; swept)
C_FALSE_BACKOUT_HOURS = 1.0      # developer hours to re-land a push backed out for an intermittent (swept)
LLM_TOKENS = 3000                # tokens for one error-line read
LLM_PRICE_PER_MTOK = 3.0         # $ per million tokens (frontier-API order of magnitude; swept)


def plat_of(p):
    p = p or ""
    if "mac" in p or "osx" in p: return "mac"
    if "windows" in p or "win" in p: return "windows"
    if "android" in p: return "android"
    if "linux" in p: return "linux"
    return "other"


def suite_of(name):
    n = (name or "").lower()
    for k in ("mochitest", "web-platform", "xpcshell", "reftest"):
        if k in n:
            return {"web-platform": "wpt"}.get(k, k)
    if "build" in n:
        return "build"
    return "other"


def dur_of(j):
    if not (j.get("end_timestamp") and j.get("start_timestamp")):
        return "mid"
    m = (j["end_timestamp"] - j["start_timestamp"]) / 60
    return "short" if m < 15 else ("mid" if m <= 40 else "long")


def minutes(j):
    return max(1.0, (j["end_timestamp"] - j["start_timestamp"]) / 60) if j.get("end_timestamp") and j.get("start_timestamp") else 25.0


def label(j):
    c = j["failure_classification_id"]
    return "regression" if c == 2 else ("intermittent" if c in (4, 7, 8, 5) else None)


def features(r):
    j = r["job"]
    newline = any(s.get("failure_new_in_rev") for s in r["suggestions"])
    known = bool(r["suggestions"]) and any((b.get("summary") or "").lower().startswith("intermittent")
                                            for b in r["suggestions"][0]["open_recent"] + r["suggestions"][0]["all_others"])
    return (plat_of(j["platform"]), dur_of(j), suite_of(j["job_type_name"]), newline, known)


def retrigger_obs(r):
    """Outcomes of other runs of the same job type on the same push (retriggers / auto-retries), excluding 'retry'."""
    j = r["job"]
    return [x["result"] for x in r["same_push_type"] if x["id"] != j["id"] and x["result"] in ("success", "testfailed", "busted")]


class Model:
    def __init__(self, train, llm=None, prior_k=2.0):
        self.prior_k = prior_k
        reg = collections.Counter(); tot = collections.Counter()
        base_reg = sum(label(r["job"]) == "regression" for r in train) / max(1, len(train))
        self.base_reg = base_reg
        for r in train:
            f = features(r); tot[f] += 1; reg[f] += label(r["job"]) == "regression"
        self.reg, self.tot = reg, tot
        # retrigger pass odds by label (pooled over suite x plat with backoff to label only)
        rp = collections.Counter(); rn = collections.Counter()
        for r in train:
            obs = retrigger_obs(r)
            if not obs:
                continue
            lab = label(r["job"]); key = (lab, suite_of(r["job"]["job_type_name"]))
            for o in obs:
                rn[key] += 1; rn[(lab,)] += 1
                if o == "success":
                    rp[key] += 1; rp[(lab,)] += 1
        self.rp, self.rn = rp, rn
        # LLM confusion
        self.llm = None
        if llm:
            cm = collections.Counter()
            for r in train:
                v = llm.get(r["job"]["id"])
                if v:
                    cm[(label(r["job"]), v)] += 1
            self.llm = cm

    def p_reg(self, f):
        """Posterior mean of P(regression | state), Beta prior centred on the train base rate (strength prior_k)."""
        a = self.base_reg * self.prior_k; b = (1 - self.base_reg) * self.prior_k
        return (self.reg[f] + a) / (self.tot[f] + a + b)

    def p_pass(self, lab, suite):
        k = (lab, suite)
        if self.rn[k] >= 10:
            return (self.rp[k] + 1) / (self.rn[k] + 2)
        return (self.rp[(lab,)] + 1) / (self.rn[(lab,)] + 2)

    def p_llm(self, lab, verdict):
        if not self.llm:
            return None
        n = sum(v for (l, _), v in self.llm.items() if l == lab)
        return (self.llm[(lab, verdict)] + 1) / (n + 2)


def costs(j, cfg):
    return {"retrigger": minutes(j) * PRICE_PER_MIN[plat_of(j["platform"])] * cfg["price_mult"],
            "llm": LLM_TOKENS / 1e6 * LLM_PRICE_PER_MTOK * cfg["llm_mult"],
            "escalate": TRIAGE_MIN / 60 * WAGE_PER_HOUR * cfg["triage_mult"],
            "miss": C_MISS_HOURS * WAGE_PER_HOUR * cfg["miss_mult"],
            "false_backout": C_FALSE_BACKOUT_HOURS * WAGE_PER_HOUR * cfg["fb_mult"]}


def solve(b, suite, c, model, depth, used_llm=False):
    """Expected cost-to-go and best action from belief b = P(regression), finite horizon `depth` information actions."""
    best = ("star", (b * c["miss"]))
    back = (1 - b) * c["false_backout"]
    if back < best[1]:
        best = ("backout", back)
    esc = c["escalate"]
    if esc < best[1]:
        best = ("escalate", esc)
    if depth > 0:
        # retrigger
        pp_r = model.p_pass("regression", suite); pp_i = model.p_pass("intermittent", suite)
        p_pass = b * pp_r + (1 - b) * pp_i
        val = c["retrigger"]
        for passed, p in ((True, p_pass), (False, 1 - p_pass)):
            if p <= 0:
                continue
            nb = (b * (pp_r if passed else 1 - pp_r)) / p
            val += p * solve(nb, suite, c, model, depth - 1, used_llm)[1]
        if val < best[1]:
            best = ("retrigger", val)
        if model.llm and not used_llm:
            val = c["llm"]
            for v in ("regression", "intermittent"):
                pr = model.p_llm("regression", v); pi = model.p_llm("intermittent", v)
                p = b * pr + (1 - b) * pi
                if p <= 0:
                    continue
                val += p * solve(b * pr / p, suite, c, model, depth - 1, True)[1]
            if val < best[1]:
                best = ("llm", val)
    return best


def simulate(r, policy, model, cfg, llm, rng_u):
    """Realised cost of a policy on one failure, using the true label; retrigger outcomes drawn from the measured odds with a
    fixed uniform per step (common random numbers across policies); LLM verdict = Qwen's actual verdict if available."""
    j = r["job"]; lab = label(j); c = costs(j, cfg); f = features(r); suite = f[2]; RT_HIST[id(r)] = []
    b = model.p_reg(f); spent = 0.0; used_llm = False; trace = []
    for step in range(cfg["depth"] + 1):
        a = policy(b, suite, c, model, cfg["depth"] - step, used_llm, r)
        trace.append(a)
        if a == "star":
            return spent + (c["miss"] if lab == "regression" else 0.0), trace
        if a == "backout":
            return spent + (c["false_backout"] if lab == "intermittent" else 0.0), trace
        if a == "escalate":
            return spent + c["escalate"], trace
        if a == "retrigger":
            spent += c["retrigger"]
            obs = retrigger_obs(r); n_rt = sum(t == "retrigger" for t in trace[:-1])
            if cfg.get("real_outcomes", True) and n_rt < len(obs):
                passed = obs[n_rt] == "success"                      # what actually happened on that push
            else:
                pp = model.p_pass(lab, suite); passed = rng_u[step] < pp
            pr, pi = model.p_pass("regression", suite), model.p_pass("intermittent", suite)
            RT_HIST.setdefault(id(r), []).append(passed)
            lr = (pr if passed else 1 - pr); li = (pi if passed else 1 - pi)
            b = b * lr / (b * lr + (1 - b) * li)
        elif a == "llm":
            spent += c["llm"]; used_llm = True
            v = llm.get(j["id"]) if llm else None
            if v:
                pr, pi = model.p_llm("regression", v), model.p_llm("intermittent", v)
                b = b * pr / (b * pr + (1 - b) * pi)
    return spent + c["escalate"], trace


def pol_dp(b, suite, c, model, depth, used_llm, r):
    return solve(b, suite, c, model, depth, used_llm)[0]


def pol_fixed(k, all_pass, all_fail, mixed, k_mac=None):
    """Rule family: retrigger up to k times (k_mac on macOS if given), then a fixed terminal move by outcome pattern."""
    def f(b, suite, c, model, depth, used_llm, r):
        n = cfgdepth - depth
        kk = k_mac if (k_mac is not None and plat_of(r["job"]["platform"]) == "mac") else k
        hist = RT_HIST.get(id(r), [])
        if n < kk:
            return "retrigger"
        if kk == 0:
            return all_pass
        if all(hist): return all_pass
        if not any(hist): return all_fail
        return mixed
    return f


def pol_llm(table):
    """Replays the LLM's own choices: first move, then its move after a retrigger passed / failed."""
    def f(b, suite, c, model, depth, used_llm, r):
        v = table.get(r["job"]["id"])
        if not v:
            return "escalate"
        n = cfgdepth - depth; hist = RT_HIST.get(id(r), [])
        if n == 0:
            return v.get("verdict") or "escalate"
        nxt = v.get("after_pass") if hist and hist[-1] else v.get("after_fail")
        if nxt == "retrigger" and n >= 2:
            return "escalate"
        return nxt or "escalate"
    return f


RT_HIST = {}
cfgdepth = 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sample"); ap.add_argument("--llm"); ap.add_argument("--split-date", default="2026-09-06")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--control"); ap.add_argument("--informed"); ap.add_argument("--write-facts"); ap.add_argument("--out")
    ap.add_argument("--ctl", action="append", default=[], help="extra LLM controller replay: NAME=path.jsonl")
    a = ap.parse_args()
    import datetime as dt, random
    R = [json.loads(x) for x in open(a.sample)]
    R = [r for r in R if label(r["job"])]
    cut = dt.datetime.fromisoformat(a.split_date).timestamp()
    train = [r for r in R if (r["job"]["submit_timestamp"] or 0) < cut]; test = [r for r in R if (r["job"]["submit_timestamp"] or 0) >= cut]
    llm = {}
    if a.llm:
        for x in open(a.llm):
            v = json.loads(x); llm[v["job_id"]] = v["verdict"]
    model = Model(train, llm or None)
    print(f"train {len(train)} test {len(test)}; train regression share {model.base_reg:.3f}; states seen {len(model.tot)}")
    print("retrigger pass odds:", {k: f"{model.rp[k]}/{model.rn[k]}" for k in model.rn if len(k) == 1})
    if model.llm:
        print("LLM confusion (truth, verdict):", dict(model.llm))
    if a.write_facts:          # numbers given to the 'informed' LLM: train window only, published prices
        c0 = costs({"platform": "linux", "start_timestamp": 0, "end_timestamp": 60}, dict(price_mult=1, llm_mult=1, triage_mult=1, miss_mult=1, fb_mult=1))
        json.dump({"price_per_min": {"mac": PRICE_PER_MIN["mac"], "osx": PRICE_PER_MIN["mac"], "windows": PRICE_PER_MIN["windows"], "win": PRICE_PER_MIN["windows"],
                                     "android": PRICE_PER_MIN["android"], "linux": PRICE_PER_MIN["linux"]},
                   "escalate": c0["escalate"], "miss": c0["miss"], "false_backout": c0["false_backout"], "base_reg": model.base_reg,
                   "pass_int": model.p_pass("intermittent", "_pooled_"), "pass_reg": model.p_pass("regression", "_pooled_")},
                  open(a.write_facts, "w"), indent=1)
        print("wrote facts", a.write_facts)
    grid = [dict(price_mult=1, llm_mult=1, triage_mult=1, miss_mult=1, fb_mult=1, depth=cfgdepth)]
    if a.sweep:                # log grid around the published prices; the best rule is re-tuned in every cell
        for key in ("price_mult", "miss_mult", "triage_mult", "fb_mult"):
            grid += [dict(grid[0], **{key: m}) for m in (0.1, 0.3, 3, 10)]
    rng = random.Random(7); U = {r["job"]["id"]: [rng.random() for _ in range(cfgdepth + 1)] for r in test}
    RULES = {}
    for k in (0, 1, 2):
        for ap_ in ("star", "escalate"):
            for af in ("backout", "escalate"):
                for mx in ("star", "escalate"):
                    if k == 0 and (af != "backout" or mx != "star"):
                        continue
                    RULES[f"k{k}:{ap_}/{af}/{mx}"] = pol_fixed(k, ap_ if k else ap_, af, mx)
                    if k:
                        RULES[f"k{k}mac0:{ap_}/{af}/{mx}"] = pol_fixed(k, ap_, af, mx, k_mac=0)
    RULES["k0:backout"] = pol_fixed(0, "backout", "backout", "backout")

    def mean_cost(pol, rows, cfg, U_):
        return sum(simulate(r, pol, model, cfg, llm, U_[r["job"]["id"]])[0] for r in rows) / max(1, len(rows))
    Utr = {r["job"]["id"]: [rng.random() for _ in range(cfgdepth + 1)] for r in train}
    llm_ctl = {}
    for nm in ("control", "informed"):
        pth = getattr(a, nm)
        if pth:
            llm_ctl[f"Qwen, {nm}"] = {json.loads(x)["job_id"]: json.loads(x) for x in open(pth)}
    for spec in a.ctl:
        nm, pth = spec.split("=", 1)
        llm_ctl[nm] = {json.loads(x)["job_id"]: json.loads(x) for x in open(pth)}
    def boot_ci(diff_by_push, n=2000):
        keys = list(diff_by_push); r = random.Random(11); out = []
        for _ in range(n):
            smp = [diff_by_push[keys[r.randrange(len(keys))]] for _ in keys]
            tot = sum(x for v in smp for x in v); cnt = sum(len(v) for v in smp)
            out.append(tot / max(1, cnt))
        out.sort(); return round(out[int(.025 * n)], 3), round(out[int(.975 * n)], 3)
    real_subset = [r for r in test if len(retrigger_obs(r)) >= cfgdepth]     # every retrigger outcome actually observed
    report = {"n_train": len(train), "n_test": len(test), "n_test_all_outcomes_observed": len(real_subset),
              "train_regression_share": model.base_reg, "retrigger_pass": {str(k): [model.rp[k], model.rn[k]] for k in model.rn if len(k) == 1},
              "llm_confusion": {str(k): v for k, v in (model.llm or {}).items()}, "cells": []}
    for cfg in grid:
        best_rule = min(RULES, key=lambda nm: mean_cost(RULES[nm], train, cfg, Utr))     # re-tuned on the TRAIN window per cost setting
        pols = {"dp": pol_dp, f"best rule ({best_rule})": RULES[best_rule], "always escalate": RULES["k0:escalate/backout/star"],
                "retrigger once, pass->star, fail->escalate": RULES["k1:star/escalate/escalate"]}
        for nm, tab in llm_ctl.items():
            pols[f"LLM controller ({nm})"] = pol_llm(tab)
        res = {}; per = {}
        for name, p in pols.items():
            tot = 0.0; miss = 0; acts = collections.Counter(); per[name] = {}
            for r in test:
                cst, tr = simulate(r, p, model, cfg, llm, U[r["job"]["id"]])
                tot += cst; acts[tuple(tr)] += 1; per[name][r["job"]["id"]] = cst
                miss += label(r["job"]) == "regression" and tr[-1] == "star"
            sub = [per[name][r["job"]["id"]] for r in real_subset]
            res[name] = (tot / max(1, len(test)), miss, acts.most_common(4), sum(sub) / max(1, len(sub)))
        cell = {"cfg": {k: v for k, v in cfg.items() if k != "depth"}, "best_rule": best_rule, "policies": {}}
        print("cfg", {k: v for k, v in cfg.items() if v != 1 and k != "depth"} or "published prices")
        for name, (m, miss, acts, msub) in res.items():
            push = collections.defaultdict(list)
            for r in test:
                push[r["job"]["push_id"]].append(per["dp"][r["job"]["id"]] - per[name][r["job"]["id"]])
            ci = boot_ci(push) if name != "dp" else None
            cell["policies"][name] = {"cost_per_failure": round(m, 3), "starred_regressions": miss, "cost_real_outcome_subset": round(msub, 3),
                                      "dp_minus_this_ci_push_clustered": ci}
            print(f"   {name:44s} ${m:7.3f}/failure  real-outcome subset ${msub:7.3f}  starred regressions {miss:4d}"
                  + (f"  DP minus this: CI {ci}" if ci else "") + f"  top paths {acts[:2]}")
        report["cells"].append(cell)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
