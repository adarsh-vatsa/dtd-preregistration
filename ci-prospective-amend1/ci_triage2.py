"""CI triage, outcome model v2 (ci-dp v2): per-test flakiness and correlated retriggers.

What changes from ci_triage.py (ci-dp v1) and nothing else (same state features, P(regression) model, costs, LLM reads, horizon):
  v1  every retrigger of a failure passes independently with one rate per (label, suite).
  v2  each failure has its own pass rate theta (how flaky that test is right now). theta ~ Beta(mu*s, (1-mu)*s) given the label,
      where mu is the test's own mean pass rate from the train window, shrunk to its job type, suite and label (hierarchical
      empirical Bayes), and s (how much theta varies around mu) is fitted on the train window by leave-one-out marginal
      likelihood. Retriggers of the same failure share theta, so they are correlated (beta-binomial); the belief is now
      (P(regression), passes, fails), still solved exactly by the same finite-horizon DP.
Everything is fitted on the train window only (before --split-date).

Evaluation (all policies scored the same way):
  real      failures whose first `depth` retrigger outcomes were all observed on Mozilla's push: real outcomes only
  sim-v1    all test failures; unobserved outcomes drawn from the v1 model (the original evaluation)
  sim-world all test failures; unobserved outcomes drawn from a beta-binomial 'world' model fitted on train + test (conditional on
            the outcomes actually observed for that failure), i.e. the best available estimate of reality, not the policy's model
Rules are re-tuned on the train window under each simulator.
usage: python ci_triage2.py <sample.jsonl> <qwen_verdicts.jsonl> [--ctl NAME=path ...] [--split-date] [--val-split-date] [--out]
"""
import argparse, collections, datetime as dt, json, math, os, random, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_triage as C


def test_path(r):
    s = r["suggestions"]
    if not s:
        return None
    parts = [p.strip() for p in s[0]["search"].split("|")]
    return parts[1] if len(parts) > 1 and parts[1] else None


def job_type(r):
    return re.sub(r"-\d+$", "", r["job"]["job_type_name"])


def keys_of(r):
    """Hierarchy of identities for the failing test, most specific first."""
    return [("path", test_path(r)), ("jt", job_type(r)), ("suite", C.suite_of(r["job"]["job_type_name"]))]


def lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def bb_loglik(seq, a, b):
    """log P(ordered pass/fail sequence) under theta ~ Beta(a, b)."""
    k = sum(seq); n = len(seq)
    return lbeta(a + k, b + n - k) - lbeta(a, b)


class OutcomeModel:
    """Hierarchical beta-binomial retrigger model, per label."""

    def __init__(self, rows, m=None, s=None, grid_m=(1, 2, 4, 8, 16), grid_s=(0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 40, 1e6)):
        self.cnt = {lab: collections.defaultdict(lambda: [0, 0]) for lab in ("intermittent", "regression")}
        self.data = []
        for r in rows:
            lab = C.label(r["job"]); seq = [x == "success" for x in C.retrigger_obs(r)]
            if not seq:
                continue
            self.data.append((lab, keys_of(r), seq))
            for k in keys_of(r) + [("label", lab)]:
                c = self.cnt[lab][k]; c[0] += sum(seq); c[1] += len(seq)
        if m is None or s is None:
            best = None
            for mm in grid_m:
                for ss in grid_s:
                    self.m, self.s = {"intermittent": mm, "regression": mm}, {"intermittent": ss, "regression": ss}
                    ll = {lab: 0.0 for lab in self.cnt}
                    for lab, ks, seq in self.data:
                        a, b = self.ab(lab, ks, exclude=seq)
                        ll[lab] += bb_loglik(seq, a, b)
                    best = best or {lab: (-1e18, None, None) for lab in self.cnt}
                    for lab in ll:
                        if ll[lab] > best[lab][0]:
                            best[lab] = (ll[lab], mm, ss)
            self.m = {lab: best[lab][1] for lab in best}; self.s = {lab: best[lab][2] for lab in best}
            self.fit_ll = {lab: best[lab][0] for lab in best}
        else:
            self.m = {"intermittent": m, "regression": m} if not isinstance(m, dict) else m
            self.s = {"intermittent": s, "regression": s} if not isinstance(s, dict) else s

    def mu(self, lab, ks, exclude=None):
        """Shrunk mean pass rate: label -> suite -> job type -> test path; `exclude` removes one failure's own outcomes."""
        ex_k, ex_n = (sum(exclude), len(exclude)) if exclude else (0, 0)
        c = self.cnt[lab][("label", lab)]
        mu = (c[0] - ex_k + 1) / (c[1] - ex_n + 2)
        for k in reversed(ks):
            if k[1] is None:
                continue
            c = self.cnt[lab].get(k)
            if not c:
                continue
            kk, nn = c[0] - ex_k, c[1] - ex_n
            if nn <= 0:
                continue
            mu = (kk + self.m[lab] * mu) / (nn + self.m[lab])
        return min(max(mu, 1e-3), 1 - 1e-3)

    def ab(self, lab, ks, exclude=None):
        mu = self.mu(lab, ks, exclude); s = self.s[lab]
        return mu * s, (1 - mu) * s

    def p_pass(self, lab, ks, hist):
        a, b = self.ab(lab, ks)
        return (a + sum(hist)) / (a + b + len(hist))


class Model2(C.Model):
    def __init__(self, train, llm=None, prior_k=2.0, outcome=None):
        super().__init__(train, llm, prior_k)
        self.om = outcome or OutcomeModel(train)


def solve2(b, ks, hist, c, model, depth, used_llm=False):
    """Same DP as ci_triage.solve, with the belief carrying this failure's retrigger history (beta-binomial predictive)."""
    best = ("star", b * c["miss"])
    back = (1 - b) * c["false_backout"]
    if back < best[1]:
        best = ("backout", back)
    if c["escalate"] < best[1]:
        best = ("escalate", c["escalate"])
    if depth > 0:
        pr = model.om.p_pass("regression", ks, hist); pi = model.om.p_pass("intermittent", ks, hist)
        p_pass = b * pr + (1 - b) * pi
        val = c["retrigger"]
        for passed, p in ((True, p_pass), (False, 1 - p_pass)):
            if p <= 0:
                continue
            nb = b * (pr if passed else 1 - pr) / p
            val += p * solve2(nb, ks, hist + [passed], c, model, depth - 1, used_llm)[1]
        if val < best[1]:
            best = ("retrigger", val)
        if model.llm and not used_llm:
            val = c["llm"]
            for v in ("regression", "intermittent"):
                lr = model.p_llm("regression", v); li = model.p_llm("intermittent", v)
                p = b * lr + (1 - b) * li
                if p <= 0:
                    continue
                val += p * solve2(b * lr / p, ks, hist, c, model, depth - 1, True)[1]
            if val < best[1]:
                best = ("llm", val)
    return best


def simulate(r, policy, world, cfg, llm, u, model_v1, model_v2, mode):
    """Realised cost of a policy on one failure. Policies get (state dict) and return an action. Belief bookkeeping for the
    v1 and v2 DPs is done here so every policy sees the same outcome stream."""
    j = r["job"]; lab = C.label(j); c = C.costs(j, cfg); f = C.features(r); suite = f[2]; ks = keys_of(r)
    obs = [x == "success" for x in C.retrigger_obs(r)]
    st = {"r": r, "c": c, "suite": suite, "ks": ks, "hist": [], "used_llm": False, "b1": model_v1.p_reg(f), "b2": model_v2.p_reg(f), "n": 0}
    spent = 0.0; trace = []
    for step in range(cfg["depth"] + 1):
        st["depth"] = cfg["depth"] - step
        a = policy(st)
        trace.append(a)
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
                passed = u[step] < model_v1.p_pass(lab, suite)
            else:   # world model, conditional on everything observed so far for this failure (real + already simulated)
                passed = u[step] < world.p_pass(lab, ks, st["hist"])
            p1r, p1i = model_v1.p_pass("regression", suite), model_v1.p_pass("intermittent", suite)
            l1r, l1i = (p1r, p1i) if passed else (1 - p1r, 1 - p1i)
            st["b1"] = st["b1"] * l1r / (st["b1"] * l1r + (1 - st["b1"]) * l1i)
            p2r, p2i = model_v2.om.p_pass("regression", ks, st["hist"]), model_v2.om.p_pass("intermittent", ks, st["hist"])
            l2r, l2i = (p2r, p2i) if passed else (1 - p2r, 1 - p2i)
            st["b2"] = st["b2"] * l2r / (st["b2"] * l2r + (1 - st["b2"]) * l2i)
            st["hist"].append(passed); C.RT_HIST[id(r)] = list(st["hist"])
        elif a == "llm":
            spent += c["llm"]; st["used_llm"] = True
            v = llm.get(j["id"]) if llm else None
            if v:
                for key, mdl in (("b1", model_v1), ("b2", model_v2)):
                    pr, pi = mdl.p_llm("regression", v), mdl.p_llm("intermittent", v)
                    st[key] = st[key] * pr / (st[key] * pr + (1 - st[key]) * pi)
    return spent + c["escalate"], trace


def mk_dp_v1(model_v1):
    return lambda st: C.solve(st["b1"], st["suite"], st["c"], model_v1, st["depth"], st["used_llm"])[0]


def mk_dp_v2(model_v2):
    return lambda st: solve2(st["b2"], st["ks"], st["hist"], st["c"], model_v2, st["depth"], st["used_llm"])[0]


def mk_rule(k, all_pass, all_fail, mixed, k_mac=None):
    def f(st):
        r = st["r"]; hist = st["hist"]; n = len(hist)
        kk = k_mac if (k_mac is not None and C.plat_of(r["job"]["platform"]) == "mac") else k
        if n < kk and not (st["used_llm"]):
            return "retrigger"
        if kk == 0:
            return all_pass
        if all(hist):
            return all_pass
        if not any(hist):
            return all_fail
        return mixed
    return f


def mk_llm_ctl(table):
    def f(st):
        v = table.get(st["r"]["job"]["id"])
        if not v:
            return "escalate"
        hist = st["hist"]; n = len(hist)
        if n == 0 and not st["used_llm"]:
            return v.get("verdict") or "escalate"
        nxt = v.get("after_pass") if hist and hist[-1] else v.get("after_fail")
        if nxt == "retrigger" and n >= 2:
            return "escalate"
        return nxt or "escalate"
    return f


def rule_family():
    R = {}
    for k in (0, 1, 2):
        for ap_ in ("star", "escalate"):
            for af in ("backout", "escalate"):
                for mx in ("star", "escalate"):
                    if k == 0 and (af != "backout" or mx != "star"):
                        continue
                    R[f"k{k}:{ap_}/{af}/{mx}"] = mk_rule(k, ap_, af, mx)
                    if k:
                        R[f"k{k}mac0:{ap_}/{af}/{mx}"] = mk_rule(k, ap_, af, mx, k_mac=0)
    R["k0:backout"] = mk_rule(0, "backout", "backout", "backout")
    return R


def boot(diff_by_push, n=2000, seed=11):
    keys = list(diff_by_push); rng = random.Random(seed); out = []
    for _ in range(n):
        smp = [diff_by_push[keys[rng.randrange(len(keys))]] for _ in keys]
        out.append(sum(x for v in smp for x in v) / max(1, sum(len(v) for v in smp)))
    out.sort(); return [round(out[int(.025 * n)], 3), round(out[int(.975 * n)], 3)]


def evaluate(train, test, llm, ctl, cfg, label_):
    model_v1 = C.Model(train, llm or None)
    model_v2 = Model2(train, llm or None)
    world = OutcomeModel(train + test)                       # simulator only; never seen by any policy
    rng = random.Random(7)
    U = {r["job"]["id"]: [rng.random() for _ in range(cfg["depth"] + 1)] for r in test + train}
    real = [r for r in test if len(C.retrigger_obs(r)) >= cfg["depth"]]
    rules = rule_family()
    out = {"label": label_, "n_train": len(train), "n_test": len(test), "n_real": len(real),
           "outcome_model_v2": {"m": model_v2.om.m, "s": model_v2.om.s},
           "world_model": {"m": world.m, "s": world.s}, "modes": {}}
    for mode in ("real", "sim-world", "sim-v1"):
        rows = real if mode == "real" else test
        trw = [r for r in train if len(C.retrigger_obs(r)) >= cfg["depth"]] if mode == "real" else train
        run = lambda pol, rr: [simulate(r, pol, world, cfg, llm, U[r["job"]["id"]], model_v1, model_v2, mode) for r in rr]
        best = min(rules, key=lambda nm: sum(x[0] for x in run(rules[nm], trw)))
        pols = {"DP v2 (per-test, correlated)": mk_dp_v2(model_v2), "DP v1 (pooled, independent)": mk_dp_v1(model_v1),
                f"best rule, re-tuned on train ({best})": rules[best], "retrigger once, pass->star, fail->escalate": rules["k1:star/escalate/escalate"],
                "always escalate": rules["k0:escalate/backout/star"]}
        for nm, tab in ctl.items():
            if all(r["job"]["id"] in tab for r in rows):
                pols[f"LLM controller ({nm})"] = mk_llm_ctl(tab)
        per = {}; res = {}
        for name, p in pols.items():
            costs_ = {}; paths = collections.Counter(); outc = collections.Counter(); mins = []
            for r in rows:
                cst, tr = simulate(r, p, world, cfg, llm, U[r["job"]["id"]], model_v1, model_v2, mode)
                costs_[r["job"]["id"]] = cst; paths[tuple(tr)] += 1; lab = C.label(r["job"])
                outc["missed regressions"] += tr[-1] == "star" and lab == "regression"
                outc["needless backouts"] += tr[-1] == "backout" and lab == "intermittent"
                outc["escalations"] += tr[-1] == "escalate"
                mins.append(C.minutes(r["job"]) * tr.count("retrigger") + 0.1 * tr.count("llm") + (C.TRIAGE_MIN if tr[-1] == "escalate" else 0))
            per[name] = costs_; mins.sort()
            res[name] = {"cost_per_failure": round(sum(costs_.values()) / len(rows), 3), **outc,
                         "median_minutes_to_settle": round(mins[len(mins) // 2], 1), "mean_minutes_to_settle": round(sum(mins) / len(mins), 1),
                         "top_paths": [[list(k), v] for k, v in paths.most_common(3)]}
        ref = "DP v2 (per-test, correlated)"
        for name in res:
            if name == ref:
                continue
            push = collections.defaultdict(list)
            for r in rows:
                push[r["job"]["push_id"]].append(per[ref][r["job"]["id"]] - per[name][r["job"]["id"]])
            res[name]["dp_v2_minus_this_ci"] = boot(push)
        out["modes"][mode] = {"n": len(rows), "best_rule": best, "policies": res}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sample"); ap.add_argument("llm"); ap.add_argument("--split-date", default="2026-09-06")
    ap.add_argument("--val-split-date", default=None, help="development check inside the train window: fit before, score after")
    ap.add_argument("--ctl", action="append", default=[]); ap.add_argument("--out")
    a = ap.parse_args()
    R = [json.loads(x) for x in open(a.sample)]; R = [r for r in R if C.label(r["job"])]
    ts = lambda r: r["job"]["submit_timestamp"] or 0
    cut = dt.datetime.fromisoformat(a.split_date).timestamp()
    train = [r for r in R if ts(r) < cut]; test = [r for r in R if ts(r) >= cut]
    if a.val_split_date:     # development: never touches the test window
        vcut = dt.datetime.fromisoformat(a.val_split_date).timestamp()
        train, test = [r for r in train if ts(r) < vcut], [r for r in train if ts(r) >= vcut]
    llm = {json.loads(x)["job_id"]: json.loads(x)["verdict"] for x in open(a.llm)}
    ctl = {}
    for spec in a.ctl:
        nm, pth = spec.split("=", 1); ctl[nm] = {json.loads(x)["job_id"]: json.loads(x) for x in open(pth)}
    cfg = dict(price_mult=1, llm_mult=1, triage_mult=1, miss_mult=1, fb_mult=1, depth=C.cfgdepth)
    t0 = time.time()
    out = evaluate(train, test, llm, ctl, cfg, "validation inside train" if a.val_split_date else "test")
    # DP v2 solve time
    m2 = Model2(train, llm); f = test[0]; ts_ = []
    for r in test[:300]:
        t = time.perf_counter(); solve2(m2.p_reg(C.features(r)), keys_of(r), [], C.costs(r["job"], cfg), m2, cfg["depth"]); ts_.append((time.perf_counter() - t) * 1e6)
    ts_.sort(); out["dp_v2_solve_microseconds"] = {"median": round(ts_[len(ts_) // 2], 1), "p90": round(ts_[int(.9 * len(ts_))], 1)}
    out["seconds"] = round(time.time() - t0, 1)
    for mode, d in out["modes"].items():
        print(f"== {mode} (n={d['n']})")
        for nm, v in sorted(d["policies"].items(), key=lambda x: x[1]["cost_per_failure"]):
            print(f"   {nm:58s} ${v['cost_per_failure']:7.3f}  miss {v['missed regressions']:3d}  needless backout {v['needless backouts']:3d}  "
                  f"esc {v['escalations']:4d}  settle med {v['median_minutes_to_settle']:5.1f}m  {v.get('dp_v2_minus_this_ci', '')}")
    print("v2 outcome model:", out["outcome_model_v2"], "world:", out["world_model"], "solve us:", out["dp_v2_solve_microseconds"])
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
