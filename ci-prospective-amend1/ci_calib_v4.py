"""CI v4 calibration (train window only). Models (v3: logistic prior + per-test beta-binomial retriggers) are fitted on data before
--fit-cut; the belief's calibration is fitted on every failure between --fit-cut and --split-date with real retrigger outcomes:
the analysis sample AND the crawl's detail rows (thousands, vs 198 sample failures used for v3t).
Calibration map (Platt scaling on the belief's components):
    logit b = a0 + alpha * prior_logit + lam_pass * (sum of log-LRs of passed retriggers) + lam_fail * (sum of log-LRs of failed
              retriggers) + kap * (log-LR of the LLM verdict)
a0, alpha, lam_pass, lam_fail fitted on sample + crawl rows; kap on sample rows only (the LLM verdicts exist only for the sample).
Writes ci_calib_v4.json. usage: python ci_calib_v4.py [--fit-cut 2026-08-10] [--split-date 2026-09-06]"""
import argparse, datetime as dt, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_triage as C, ci_triage2 as T, ci_history as H, ci_triage3 as T3

ap = argparse.ArgumentParser(); ap.add_argument("--fit-cut", default="2026-08-10"); ap.add_argument("--split-date", default="2026-09-06")
ap.add_argument("--out", default="ci_calib_v4.json"); a = ap.parse_args()
ts = lambda r: r["job"]["submit_timestamp"] or 0
fcut = dt.datetime.fromisoformat(a.fit_cut).timestamp(); cut = dt.datetime.fromisoformat(a.split_date).timestamp()
S = [json.loads(x) for x in open(f"{H.D}/sample/sample.jsonl")]; S = [r for r in S if C.label(r["job"])]; sids = {r["job"]["id"] for r in S}
J = H.load_jobs(); push = H.PushIndex(J)
llm = {json.loads(x)["job_id"]: json.loads(x)["verdict"] for x in open("ci_llm_verdict.jsonl")}
fit_s = [r for r in S if ts(r) < fcut]
crawl_fit = H.load_history(J, fcut, sids)
dps, v1, meta = T3.build(fit_s, crawl_fit, llm, J, push)
d3 = [d for d in dps if d.name.startswith("v3:")][0]
crawl_cal = [r for r in H.load_history(J, cut, sids) if ts(r) >= fcut and r.get("runs_fetched")]
cal_s = [r for r in S if fcut <= ts(r) < cut]
rows = [(r, "sample") for r in cal_s if C.retrigger_obs(r)] + [(r, "crawl") for r in crawl_cal if C.retrigger_obs(r)]


def comps(r, k=2):
    obs = [x == "success" for x in C.retrigger_obs(r)][:k]; zp = zf = 0.0; hist = []
    for o in obs:
        pr, pi = d3.pp("regression", r, hist), d3.pp("intermittent", r, hist)
        if o:
            zp += math.log(pr / pi)
        else:
            zf += math.log((1 - pr) / (1 - pi))
        hist.append(o)
    v = llm.get(r["job"]["id"]); zl = math.log(v1.p_llm("regression", v) / v1.p_llm("intermittent", v)) if v else 0.0
    p = min(max(d3._prior(r), 1e-6), 1 - 1e-6)
    return [1.0, math.log(p / (1 - p)), zp, zf], zl


Z, ZL, Y, SRC = [], [], [], []
for r, src in rows:
    z, zl = comps(r); Z.append(z); ZL.append(zl); Y.append(C.label(r["job"]) == "regression"); SRC.append(src)
Z = np.array(Z); ZL = np.array(ZL); Y = np.array(Y, float); SRC = np.array(SRC)
lr = H.Logistic(l2=1e-3).fit(Z, Y)
w = lr.w
# kappa on sample rows, holding the other weights fixed (1-D grid by log loss)
ms = SRC == "sample"
def ll_k(k):
    p = 1 / (1 + np.exp(-(Z[ms] @ w + k * ZL[ms]))); p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(Y[ms] * np.log(p) + (1 - Y[ms]) * np.log(1 - p)).mean())
kap = min(np.arange(0, 2.01, 0.05), key=ll_k)
def ll(wv, k, mask):
    p = 1 / (1 + np.exp(-(Z[mask] @ wv + k * ZL[mask]))); p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(Y[mask] * np.log(p) + (1 - Y[mask]) * np.log(1 - p)).mean())
allm = np.ones(len(Y), bool)
res = {"n_rows": int(len(Y)), "n_sample": int(ms.sum()), "n_crawl": int((~ms).sum()),
       "a0": float(w[0]), "alpha": float(w[1]), "lam_pass": float(w[2]), "lam_fail": float(w[3]), "kap": float(kap),
       "logloss_uncalibrated_all": ll(np.array([0, 1, 1, 1.0]), 0.0, allm), "logloss_calibrated_all": ll(w, 0.0, allm),
       "logloss_uncalibrated_sample_with_llm": ll(np.array([0, 1, 1, 1.0]), 1.0, ms), "logloss_calibrated_sample_with_llm": ll(w, kap, ms),
       "v3t_on_same_rows_sample_with_llm": None}
# calibration table after calibration, belief bins
p = 1 / (1 + np.exp(-(Z @ w))); bins = {}
for lo in (0, .2, .4, .6, .75, .9):
    hi = {0: .2, .2: .4, .4: .6, .6: .75, .75: .9, .9: 1.01}[lo]; m = (p >= lo) & (p < hi)
    bins[f"{lo:.2f}-{min(hi,1):.2f}"] = f"{int(Y[m].sum())}/{int(m.sum())}"
res["calibrated_bins_actual_reg_over_n"] = bins
p0 = 1 / (1 + np.exp(-(Z @ np.array([0, 1, 1, 1.0])))); bins0 = {}
for lo in (0, .2, .4, .6, .75, .9):
    hi = {0: .2, .2: .4, .4: .6, .6: .75, .75: .9, .9: 1.01}[lo]; m = (p0 >= lo) & (p0 < hi)
    bins0[f"{lo:.2f}-{min(hi,1):.2f}"] = f"{int(Y[m].sum())}/{int(m.sum())}"
res["uncalibrated_bins_actual_reg_over_n"] = bins0
print(json.dumps(res, indent=1))
json.dump(res, open(a.out, "w"), indent=1)
