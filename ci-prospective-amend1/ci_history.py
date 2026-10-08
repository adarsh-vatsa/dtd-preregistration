"""CI triage, v3 inputs: use all the real history Mozilla's crawl gives us, not just the 5,000-failure analysis sample.

  history rows   the crawl's 'detail' stage (40% of tier-1/2 failed jobs: bug suggestions; 20%: same-push runs) joined with the
                 job metadata and staff labels -> same record format as the sample; used for fitting only (train window)
  push index     every failed job in the 120-day crawl (223,770), by push: lets a decision use how many other jobs on the same push
                 had already failed, and whether the same job type had already failed on another platform (strictly earlier end
                 times, so it is known when the decision is made)
  P(regression)  L2 logistic regression (numpy) on: platform, duration, suite, new-line flag, open-intermittent-bug flag,
                 'Frequent' bug flag, push failure counts, and shrunk label history of the failing test path and job type
"""
import ast, collections, glob, json, math, os, re, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_triage as C
import ci_triage2 as T

D = os.path.expanduser("~/dtd-bounded/data/treeherder")


def _parse(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return []
    return v or []


def load_jobs(repo="autoland"):
    J = {}
    for f in sorted(glob.glob(f"{D}/jobs/{repo}/*.jsonl")):
        for x in open(f):
            j = json.loads(x); J[j["id"]] = j
    return J


def load_history(J, before_ts, exclude_ids=(), repo="autoland"):
    """Detail-stage records (tier 1, staff-labelled, submitted before `before_ts`) in the sample's record format."""
    out = []
    for f in sorted(glob.glob(f"{D}/detail/{repo}/*.jsonl")):
        for x in open(f):
            d = json.loads(x); jid = int(d["job_id"])
            j = J.get(jid)
            if not j or j["tier"] != 1 or not C.label(j) or jid in exclude_ids or (j["submit_timestamp"] or 0) >= before_ts:
                continue
            spt = _parse(d.get("same_push_type"))
            out.append({"job": j, "suggestions": _parse(d.get("suggestions")) or [], "same_push_type": spt, "errors": [], "log": None,
                        "runs_fetched": int(jid) % 10 < 2})
    return out


class PushIndex:
    def __init__(self, J):
        self.by_push = collections.defaultdict(list)
        for j in J.values():
            if j.get("end_timestamp"):
                self.by_push[j["push_id"]].append((j["end_timestamp"], T.job_type({"job": j}), C.plat_of(j["platform"]), j["id"]))
        for v in self.by_push.values():
            v.sort()

    def feats(self, j):
        t = j.get("end_timestamp") or j.get("submit_timestamp") or 0
        jt = T.job_type({"job": j}); pl = C.plat_of(j["platform"])
        before = [x for x in self.by_push.get(j["push_id"], []) if x[0] < t and x[3] != j["id"]]
        same_jt = [x for x in before if x[1] == jt]
        return {"push_fail_before": len(before), "push_jt_before": len({x[1] for x in before}),
                "same_jt_before": len(same_jt), "same_jt_other_plat_before": len({x[2] for x in same_jt if x[2] != pl})}


def bug_flags(r):
    s = r["suggestions"]
    if not s:
        return {"has_sugg": 0, "known": 0, "frequent": 0, "newline": 0, "n_sugg": 0, "no_bug": 1}
    bugs = s[0].get("open_recent", []) + s[0].get("all_others", [])
    summ = [(b.get("summary") or "").lower() for b in bugs]
    return {"has_sugg": 1, "known": int(any(x.startswith("intermittent") for x in summ)), "frequent": int(any(x.startswith("frequent") or x.startswith("perma") for x in summ)),
            "newline": int(any(x.get("failure_new_in_rev") for x in s)), "n_sugg": min(len(s), 10), "no_bug": int(not bugs)}


class LabelHistory:
    """Shrunk P(regression) of the failing test path / job type / job signature, from fitting rows."""

    def __init__(self, rows, m=16):
        self.m = m; self.cnt = collections.defaultdict(lambda: [0, 0]); n = 0; reg = 0
        for r in rows:
            y = C.label(r["job"]) == "regression"; n += 1; reg += y
            for k in self.keys(r):
                if k[1] is not None:
                    self.cnt[k][0] += y; self.cnt[k][1] += 1
        self.base = reg / max(1, n)

    @staticmethod
    def keys(r):
        return [("jt", T.job_type(r)), ("path", T.test_path(r))]

    def logit_feats(self, r, exclude_self=False):
        out = {}; q = self.base
        for k in self.keys(r):
            c = self.cnt.get(k, [0, 0]); kk, nn = c
            if exclude_self and nn:
                kk -= C.label(r["job"]) == "regression"; nn -= 1
            q = (kk + self.m * q) / (nn + self.m)
            out[f"hist_{k[0]}"] = math.log(q / (1 - q)); out[f"hist_{k[0]}_n"] = math.log1p(nn)
        return out


PLATS = ("linux", "windows", "mac", "android", "other"); DURS = ("short", "mid", "long")
SUITES = ("mochitest", "wpt", "xpcshell", "reftest", "build", "other")


def featurize(r, push, lh, exclude_self=False):
    j = r["job"]; f = C.features(r); b = bug_flags(r); p = push.feats(j); h = lh.logit_feats(r, exclude_self)
    x = [1.0] + [float(f[0] == v) for v in PLATS[1:]] + [float(f[1] == v) for v in DURS[1:]] + [float(f[2] == v) for v in SUITES[1:]]
    x += [b["has_sugg"], b["known"], b["frequent"], b["newline"], b["n_sugg"] / 10, b["no_bug"]]
    x += [math.log1p(p["push_fail_before"]), math.log1p(p["push_jt_before"]), math.log1p(p["same_jt_before"]), math.log1p(p["same_jt_other_plat_before"]),
          float(p["same_jt_other_plat_before"] > 0)]
    x += [h["hist_jt"], h["hist_jt_n"], h["hist_path"], h["hist_path_n"], float(j.get("platform_option") == "debug")]
    return x


class Logistic:
    def __init__(self, l2=1.0, iters=50):
        self.l2, self.iters = l2, iters

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float); w = np.zeros(X.shape[1])
        R = np.eye(X.shape[1]) * self.l2; R[0, 0] = 0
        for _ in range(self.iters):            # Newton steps
            p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) + R @ w
            H = (X * (p * (1 - p))[:, None]).T @ X + R
            step = np.linalg.solve(H, g); w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w; return self

    def predict(self, X):
        return 1 / (1 + np.exp(-np.asarray(X, float) @ self.w))


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
