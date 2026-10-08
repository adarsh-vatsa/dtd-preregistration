"""CI triage: ci-dp v1 vs v2 vs v3, same decision procedure (finite-horizon belief DP over star / backout / retrigger / LLM read /
escalate, depth 2, same published costs, same Qwen verdicts as the LLM-read observation). Only the odds change:

  v1  P(regression) from the 5-feature cell table (sample train); retriggers independent, one pass rate per (label, suite)
  v2  v1's P(regression); retriggers beta-binomial with per-test flakiness (ci_triage2.OutcomeModel), fitted on the sample
  v3  P(regression) from a logistic model with the failing test's label history and push signals (ci_history), fitted on the
      sample train window with label history from the full crawl train window; retriggers beta-binomial fitted on sample + crawl

Scoring modes as in ci_triage2: real (fully observed failures, real outcomes only), sim-world (unobserved outcomes from a
beta-binomial fitted on all data, including test, conditional on observed outcomes), sim-v1 (the original simulator).
usage: python ci_triage3.py --llm ci_llm_verdict.jsonl [--val-cut DATE (development)] [--ctl NAME=path] [--out f.json]
"""
import argparse, collections, datetime as dt, json, math, os, random, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_triage as C, ci_triage2 as T, ci_history as H

M_HIST, L2 = 64, 30.0          # chosen on the validation window inside train (ci_v3_dev.py); fixed before any test scoring
# evidence tempering (power likelihood) per model: prior logit x alpha, retrigger log-LR x lam, LLM log-LR x kap; fitted by log loss
# of the belief on validation failures with real retrigger outcomes (ci_calib_dev.py, fit before 2026-08-10, scored 08-10..09-06)
TEMPER = {"v1": (0.9, 0.6, 1.5), "v2": (0.9, 0.7, 0.75), "v3": (0.75, 0.5, 1.5)}


class DP:
    """Belief DP with a P(regression) prior and a retrigger outcome model; `om` None = v1 pooled independent outcomes."""

    def __init__(self, name, prior, cellmodel, om=None, temper=(1.0, 1.0, 1.0)):
        self.name, self.cm, self.om = name, cellmodel, om
        self.al, self.lam, self.kap = temper; self._prior = prior

    def prior(self, r):
        p = min(max(self._prior(r), 1e-6), 1 - 1e-6)
        return 1 / (1 + math.exp(-self.al * math.log(p / (1 - p))))

    def _post(self, b, lr, li, w):
        """belief update with a tempered likelihood ratio (lr/li)^w"""
        b = min(max(b, 1e-9), 1 - 1e-9)
        z = math.log(b / (1 - b)) + w * (math.log(lr) - math.log(li))
        return 1 / (1 + math.exp(-z))

    def pp(self, lab, r, hist):
        if self.om is None:
            return self.cm.p_pass(lab, C.features(r)[2])
        return self.om.p_pass(lab, T.keys_of(r), hist)

    def solve(self, b, r, hist, c, depth, used_llm):
        best = ("star", b * c["miss"])
        if (1 - b) * c["false_backout"] < best[1]:
            best = ("backout", (1 - b) * c["false_backout"])
        if c["escalate"] < best[1]:
            best = ("escalate", c["escalate"])
        if depth > 0:
            pr, pi = self.pp("regression", r, hist), self.pp("intermittent", r, hist)
            p_pass = b * pr + (1 - b) * pi; val = c["retrigger"]
            for passed, p in ((True, p_pass), (False, 1 - p_pass)):
                if p > 0:
                    nb = self._post(b, pr if passed else 1 - pr, pi if passed else 1 - pi, self.lam)
                    val += p * self.solve(nb, r, hist + [passed], c, depth - 1, used_llm)[1]
            if val < best[1]:
                best = ("retrigger", val)
            if self.cm.llm and not used_llm:
                val = c["llm"]
                for v in ("regression", "intermittent"):
                    lr, li = self.cm.p_llm("regression", v), self.cm.p_llm("intermittent", v)
                    p = b * lr + (1 - b) * li
                    if p > 0:
                        val += p * self.solve(self._post(b, lr, li, self.kap), r, hist, c, depth - 1, True)[1]
                if val < best[1]:
                    best = ("llm", val)
        return best

    def upd_rt(self, b, r, hist, passed):
        pr, pi = self.pp("regression", r, hist), self.pp("intermittent", r, hist)
        lr, li = (pr, pi) if passed else (1 - pr, 1 - pi)
        return self._post(b, lr, li, self.lam)

    def upd_llm(self, b, v):
        return self._post(b, self.cm.p_llm("regression", v), self.cm.p_llm("intermittent", v), self.kap)


class DP4(DP):
    """v4t: v3's models with the belief recalibrated (Platt map fitted on train-window failures with real outcomes, sample + crawl;
    ci_calib_v4.py): logit b = a0 + alpha*prior_logit + lam_pass*sum(passed log-LR) + lam_fail*sum(failed log-LR) + kap*LLM log-LR."""

    def __init__(self, name, prior, cellmodel, om, cal):
        super().__init__(name, prior, cellmodel, om)
        self.a0, self.al, self.lp, self.lf, self.kap = cal["a0"], cal["alpha"], cal["lam_pass"], cal["lam_fail"], cal["kap"]

    def prior(self, r):
        p = min(max(self._prior(r), 1e-6), 1 - 1e-6)
        return 1 / (1 + math.exp(-(self.a0 + self.al * math.log(p / (1 - p)))))

    def solve(self, b, r, hist, c, depth, used_llm):
        best = ("star", b * c["miss"])
        if (1 - b) * c["false_backout"] < best[1]:
            best = ("backout", (1 - b) * c["false_backout"])
        if c["escalate"] < best[1]:
            best = ("escalate", c["escalate"])
        if depth > 0:
            pr, pi = self.pp("regression", r, hist), self.pp("intermittent", r, hist)
            nb_pass = self._post(b, pr, pi, self.lp); nb_fail = self._post(b, 1 - pr, 1 - pi, self.lf)
            # predictive P(pass) under the calibrated belief
            p_pass = b * pr + (1 - b) * pi; val = c["retrigger"]
            val += p_pass * self.solve(nb_pass, r, hist + [True], c, depth - 1, used_llm)[1]
            val += (1 - p_pass) * self.solve(nb_fail, r, hist + [False], c, depth - 1, used_llm)[1]
            if val < best[1]:
                best = ("retrigger", val)
            if self.cm.llm and not used_llm:
                val = c["llm"]
                for v in ("regression", "intermittent"):
                    lr, li = self.cm.p_llm("regression", v), self.cm.p_llm("intermittent", v)
                    p = b * lr + (1 - b) * li
                    if p > 0:
                        val += p * self.solve(self._post(b, lr, li, self.kap), r, hist, c, depth - 1, True)[1]
                if val < best[1]:
                    best = ("llm", val)
        return best

    def upd_rt(self, b, r, hist, passed):
        pr, pi = self.pp("regression", r, hist), self.pp("intermittent", r, hist)
        return self._post(b, pr, pi, self.lp) if passed else self._post(b, 1 - pr, 1 - pi, self.lf)


def simulate(r, policy, dps, world, cfg, llm, u, mode, v1cm):
    j = r["job"]; lab = C.label(j); c = C.costs(j, cfg)
    obs = [x == "success" for x in C.retrigger_obs(r)]
    st = {"r": r, "c": c, "hist": [], "used_llm": False, "b": {d.name: d.prior(r) for d in dps}}
    spent = 0.0; trace = []
    for step in range(cfg["depth"] + 2):
        st["depth"] = max(0, cfg["depth"] - step); st["left"] = cfg["depth"] - step
        a = policy(st); trace.append(a)
        if a == "star":
            return spent + (c["miss"] if lab == "regression" else 0.0), trace
        if a == "backout":
            return spent + (c["false_backout"] if lab == "intermittent" else 0.0), trace
        if a == "escalate":
            return spent + c["escalate"], trace
        if a == "retrigger":
            spent += c["retrigger"]; i = len(st["hist"])
            if i < len(obs):
                passed = obs[i]
            elif mode == "sim-v1":
                passed = u[step] < v1cm.p_pass(lab, C.features(r)[2])
            else:
                passed = u[step] < world.p_pass(lab, T.keys_of(r), st["hist"])
            for d in dps:
                st["b"][d.name] = d.upd_rt(st["b"][d.name], r, st["hist"], passed)
            st["hist"].append(passed)
        elif a == "llm":
            spent += c["llm"]; st["used_llm"] = True
            v = llm.get(j["id"])
            if v:
                for d in dps:
                    st["b"][d.name] = d.upd_llm(st["b"][d.name], v)
    return spent + c["escalate"], trace


def pol_dp(d, extra_depth=0):
    """POMDP: finite-horizon DP over beliefs (values information). extra_depth > 0 allows more retriggers / reads than depth 2
    (only meaningful in simulated modes)."""
    return lambda st: d.solve(st["b"][d.name], st["r"], st["hist"], st["c"], max(0, st["left"] + extra_depth), st["used_llm"])[0]


def pol_qmdp(d):
    """QMDP: expected cost of each move as if the label became known right after it; information is worthless, so it picks the
    cheapest terminal move (star / backout / escalate) under the current belief."""
    return lambda st: d.solve(st["b"][d.name], st["r"], st["hist"], st["c"], 0, st["used_llm"])[0]


def pol_mdp(d, first=None):
    """MDP on the top guess: act as if the more likely label were true (backout if P(regression) >= 0.5, else star); optionally
    take one information move first ('llm' or 'retrigger')."""
    def f(st):
        if first == "llm" and not st["used_llm"]:
            return "llm"
        if first == "retrigger" and not st["hist"]:
            return "retrigger"
        return "backout" if st["b"][d.name] >= 0.5 else "star"
    return f


def rule(k, all_pass, all_fail, mixed, k_mac=None):
    def f(st):
        hist = st["hist"]; kk = k_mac if (k_mac is not None and C.plat_of(st["r"]["job"]["platform"]) == "mac") else k
        if len(hist) < kk:
            return "retrigger"
        if kk == 0 or all(hist):
            return all_pass
        return all_fail if not any(hist) else mixed
    return f


def rules():
    R = {}
    for k in (0, 1, 2):
        for ap_ in ("star", "escalate"):
            for af in ("backout", "escalate"):
                for mx in ("star", "escalate"):
                    if k == 0 and (af != "backout" or mx != "star"):
                        continue
                    R[f"k{k}:{ap_}/{af}/{mx}"] = rule(k, ap_, af, mx)
                    if k:
                        R[f"k{k}mac0:{ap_}/{af}/{mx}"] = rule(k, ap_, af, mx, k_mac=0)
    R["k0:backout"] = rule(0, "backout", "backout", "backout")
    return R


def llm_ctl(table):
    def f(st):
        v = table.get(st["r"]["job"]["id"])
        if not v:
            return "escalate"
        hist = st["hist"]
        if not hist:
            return v.get("verdict") or "escalate"
        nxt = v.get("after_pass") if hist[-1] else v.get("after_fail")
        return "escalate" if (nxt == "retrigger" and len(hist) >= 2) else (nxt or "escalate")
    return f


def build(train_s, crawl_fit, llm, J, push):
    v1 = C.Model(train_s, llm)
    om2 = T.OutcomeModel(train_s)
    om3 = T.OutcomeModel(train_s + [r for r in crawl_fit if r.get("runs_fetched", True)])
    lh = H.LabelHistory(train_s + crawl_fit, m=M_HIST)
    X = [H.featurize(r, push, lh, exclude_self=True) for r in train_s]; y = [C.label(r["job"]) == "regression" for r in train_s]
    lr = H.Logistic(l2=L2).fit(X, y)
    cache = {}

    def p3(r):
        k = r["job"]["id"]
        if k not in cache:
            cache[k] = float(lr.predict([H.featurize(r, push, lh)])[0])
        return cache[k]
    p1 = lambda r: v1.p_reg(C.features(r))
    dps = []
    if os.path.exists(os.environ.get("CI_CAL_V4", "ci_calib_v4.json")):
        dps.append(DP4("v4t: v3 models + belief recalibrated on sample+crawl outcomes", p3, v1, om3, json.load(open(os.environ.get("CI_CAL_V4", "ci_calib_v4.json")))))
    dps += [DP("v3t: v3 + calibrated evidence", p3, v1, om3, TEMPER["v3"]),
           DP("v3: crawl history + logistic prior + per-test correlated retriggers", p3, v1, om3),
           DP("v2t: v2 + calibrated evidence", p1, v1, om2, TEMPER["v2"]),
           DP("v2: per-test correlated retriggers", p1, v1, om2),
           DP("v1t: v1 + calibrated evidence", p1, v1, None, TEMPER["v1"]),
           DP("v1: as pre-registered", p1, v1, None)]
    return dps, v1, {"om2": [om2.m, om2.s], "om3": [om3.m, om3.s], "logistic_w": [round(float(w), 3) for w in lr.w]}


def boot(diff, n=2000, seed=11):
    keys = list(diff); rng = random.Random(seed); out = []
    for _ in range(n):
        smp = [diff[keys[rng.randrange(len(keys))]] for _ in keys]
        out.append(sum(x for v in smp for x in v) / max(1, sum(len(v) for v in smp)))
    out.sort(); return [round(out[int(.025 * n)], 3), round(out[int(.975 * n)], 3)]


def evaluate(train_s, test, crawl_fit, crawl_all, llm, ctl, J, push, cfg):
    dps, v1, meta = build(train_s, crawl_fit, llm, J, push)
    world = T.OutcomeModel(train_s + test + [r for r in crawl_all if r.get("runs_fetched", True)])
    rng = random.Random(7); U = {r["job"]["id"]: [rng.random() for _ in range(cfg["depth"] + 2)] for r in train_s + test}
    real = [r for r in test if len(C.retrigger_obs(r)) >= cfg["depth"]]
    RULES = rules(); out = {"n_train": len(train_s), "n_crawl_fit": len(crawl_fit), "n_test": len(test), "n_real": len(real),
                            "models": meta, "world": [world.m, world.s], "modes": {}}
    ref = "POMDP " + dps[0].name
    for mode in ("real", "sim-world", "sim-v1"):
        rows = real if mode == "real" else test
        trw = [r for r in train_s if len(C.retrigger_obs(r)) >= cfg["depth"]] if mode == "real" else train_s
        run = lambda pol, rr: sum(simulate(r, pol, dps, world, cfg, llm, U[r["job"]["id"]], mode, v1)[0] for r in rr)
        best = min(RULES, key=lambda nm: run(RULES[nm], trw))
        pols = {f"POMDP {d.name}": pol_dp(d) for d in dps}
        for d in dps:
            if d.name.startswith(("v4t", "v3t", "v1:")):
                pols[f"QMDP {d.name}"] = pol_qmdp(d)
                for fst in (None, "llm", "retrigger"):
                    pols[f"MDP top guess{' after ' + fst if fst else ''} {d.name}"] = pol_mdp(d, fst)
                if mode != "real":
                    pols[f"POMDP depth 3 {d.name}"] = pol_dp(d, 1)
        pols.update({f"best rule, re-tuned on train ({best})": RULES[best], "retrigger once, pass->star, fail->escalate": RULES["k1:star/escalate/escalate"],
                     "always escalate": RULES["k0:escalate/backout/star"]})
        for nm, tab in ctl.items():
            if all(r["job"]["id"] in tab for r in rows):
                pols[f"LLM controller ({nm})"] = llm_ctl(tab)
        per = {}; res = {}
        for name, p in pols.items():
            cs = {}; paths = collections.Counter(); oc = collections.Counter(); mins = []
            for r in rows:
                cst, tr = simulate(r, p, dps, world, cfg, llm, U[r["job"]["id"]], mode, v1)
                cs[r["job"]["id"]] = cst; paths[tuple(tr)] += 1; lab = C.label(r["job"])
                oc["missed regressions"] += tr[-1] == "star" and lab == "regression"
                oc["needless backouts"] += tr[-1] == "backout" and lab == "intermittent"
                oc["escalations"] += tr[-1] == "escalate"; oc["llm reads"] += tr.count("llm"); oc["retriggers"] += tr.count("retrigger")
                mins.append(C.minutes(r["job"]) * tr.count("retrigger") + 0.1 * tr.count("llm") + (C.TRIAGE_MIN if tr[-1] == "escalate" else 0))
            per[name] = cs; mins.sort()
            res[name] = {"cost_per_failure": round(sum(cs.values()) / len(rows), 3), **oc,
                         "median_minutes_to_settle": round(mins[len(mins) // 2], 1), "mean_minutes_to_settle": round(sum(mins) / len(mins), 1),
                         "top_paths": [[list(k), v] for k, v in paths.most_common(3)]}
        for name in res:
            if name != ref:
                push_d = collections.defaultdict(list)
                for r in rows:
                    push_d[r["job"]["push_id"]].append(per[ref][r["job"]["id"]] - per[name][r["job"]["id"]])
                res[name]["ref_minus_this_ci"] = boot(push_d)
        out["modes"][mode] = {"n": len(rows), "best_rule": best, "policies": res}
    t = []
    for r in test[:300]:
        b = dps[0].prior(r); c = C.costs(r["job"], cfg); t0 = time.perf_counter(); dps[0].solve(b, r, [], c, cfg["depth"], False); t.append((time.perf_counter() - t0) * 1e6)
    t.sort(); out["dp_v3_solve_microseconds"] = {"median": round(t[len(t) // 2], 1), "p90": round(t[int(.9 * len(t))], 1)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--llm", default="ci_llm_verdict.jsonl"); ap.add_argument("--split-date", default="2026-09-06")
    ap.add_argument("--val-cut"); ap.add_argument("--ctl", action="append", default=[]); ap.add_argument("--out")
    a = ap.parse_args()
    ts = lambda r: r["job"]["submit_timestamp"] or 0
    cut = dt.datetime.fromisoformat(a.split_date).timestamp()
    S = [json.loads(x) for x in open(f"{H.D}/sample/sample.jsonl")]; S = [r for r in S if C.label(r["job"])]
    sids = {r["job"]["id"] for r in S}
    J = H.load_jobs(); push = H.PushIndex(J)
    if a.val_cut:                       # development: fit before val-cut, score val-cut..split-date; test window untouched
        vcut = dt.datetime.fromisoformat(a.val_cut).timestamp()
        train_s = [r for r in S if ts(r) < vcut]; test = [r for r in S if vcut <= ts(r) < cut]
        crawl_fit = H.load_history(J, vcut, sids); crawl_all = H.load_history(J, cut, sids)
    else:
        train_s = [r for r in S if ts(r) < cut]; test = [r for r in S if ts(r) >= cut]
        crawl_fit = H.load_history(J, cut, sids); crawl_all = H.load_history(J, 1e12, sids)
    llm = {json.loads(x)["job_id"]: json.loads(x)["verdict"] for x in open(a.llm)}
    ctl = {}
    for spec in a.ctl:
        nm, pth = spec.split("=", 1); ctl[nm] = {json.loads(x)["job_id"]: json.loads(x) for x in open(pth)}
    cfg = dict(price_mult=1, llm_mult=1, triage_mult=1, miss_mult=1, fb_mult=1, depth=C.cfgdepth)
    out = evaluate(train_s, test, crawl_fit, crawl_all, llm, ctl, J, push, cfg)
    out["window"] = "validation inside train" if a.val_cut else "test"
    for mode, d in out["modes"].items():
        print(f"== {mode} (n={d['n']})")
        for nm, v in sorted(d["policies"].items(), key=lambda x: x[1]["cost_per_failure"]):
            print(f"   {nm[:70]:70s} ${v['cost_per_failure']:7.3f} miss {v['missed regressions']:3d} backout- {v['needless backouts']:3d} esc {v['escalations']:4d} "
                  f"settle {v['median_minutes_to_settle']:5.1f}m {v.get('ref_minus_this_ci', '')}")
    print(json.dumps({k: out[k] for k in ("models", "world", "dp_v3_solve_microseconds")}))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
