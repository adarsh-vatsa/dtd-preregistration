# Discretize, Then Decide: pre-registrations

Frozen forecasts, posted before the data they predict exists. Each folder holds the exact files that were frozen, and their
SHA-256 hashes are listed below so anyone can check that nothing changed after posting.

## `ci-prospective/`: CI failure triage on Mozilla's public CI (frozen 2026-10-07 09:18:41 UTC)

**Question.** When a test job fails on Mozilla's `autoland` branch, which move costs least: retrigger it, have an LLM read the
error lines, escalate to a developer, mark it intermittent, or back out the push? A finite-horizon dynamic program (DP) over
odds measured from Mozilla's own history is compared with (a) the best fixed rule, re-tuned on the same history, and (b) LLM
controllers (Qwen3.8-27B and Claude Opus 5.5), with and without the price list and measured odds in their prompt.

**History used for the freeze.** 5,000 labelled tier-1 failed jobs from Mozilla Treeherder, the latest submitted
2026-10-05 22:46 UTC. Labels are Mozilla's own failure classifications (fixed by commit, intermittent, infra).

**Test window (not yet happened at the freeze).**
- Week 1: from the freeze to 2026-10-14 00:00 UTC, at published prices.
- Week 2: 2026-10-14 00:00 UTC to 2026-10-21 00:00 UTC, after a price change announced now: **macOS runner minutes cost 5×
  and Windows 2× the published price**; everything else unchanged. The DP re-solves with no new data; the rule is re-tuned on
  the frozen history under the new prices; the informed LLM gets the new price list.
- Jobs: tier-1 failed jobs on `autoland` with a staff label by 2026-10-23; deterministic 20% sample (job id ending in 0 or 1),
  capped at 3,000 per week.

**Forecasts (cost per failure, 95% interval clustered by push).**

| | DP | best rule (re-tuned) | DP minus rule |
|---|---|---|---|
| Week 1, published prices | $8.15 [6.98, 9.54] | $13.18 [11.98, 14.51] (`k2:star/backout/escalate`) | −$5.04 [−6.51, −3.44] |
| Week 2, macOS ×5, Windows ×2 | $9.04 [7.77, 10.45] | $15.31 [13.85, 16.82] (`k1:star/escalate/star`) | −$6.27 [−7.95, −4.66] |

The forecasts are computed on the 1,176 failures in the 30 days before the freeze, which the frozen model was also fitted on,
so they may be optimistic; the test will show.

**Pre-registered predictions.**
- P1: the DP's cost per failure in each week falls inside its forecast interval.
- P2: DP minus best rule is below zero in each week (95% interval below zero).
- P3: DP minus each LLM controller is below zero in each week.
- P4: after the price change the DP's share of macOS retriggers falls, and its forecast still holds.

**Scoring** (after 2026-10-23): `ci_triage.simulate` with the frozen model; real same-push retrigger outcomes when observed,
otherwise the frozen odds; costs at the regime's prices. Prices: GitHub-hosted runner list prices ($0.006/min Linux,
$0.010 Windows, $0.062 macOS); developer time $64.44/h (US BLS median); 15 minutes per investigation; a missed regression
costs 2 developer-hours and a needless backout 1 hour.

**Files and SHA-256.**

| file | sha256 |
|---|---|
| `PREREG_ci_prospective.json` | `541e2ae4004e1a839c1b8dca1210697849be588f399ffdf90a9f92e84a77dbd1` |
| `ci_model_frozen.json` | `c7fd7a96f1c7d7f7847410ff8faa1922673e3a425a305787c128102c3673a1b9` |
| `ci_triage.py` | `b9576d1b2552a4a407ec39dccc2352e7df5a52d624df60bd614f2b23412b592f` |
| `ci_llm.py` (LLM prompts) | `a7f86bb7b65d92b64a2ce6342cfda3d8ae338c914ddc0e6bc891cc54328fdfcd` |
| `ci_prospective.py` | `b95d75fe28289eb4e8621099e0772daa28f33c12026f1b5d11a3a2b8e5b1389a` |
| `ci_facts.json` (numbers given to the informed LLM) | `45eef9bd2a27ccb2c09201229406877fe237b10b8672643f1a779436882527c6` |

Check with `shasum -a 256 ci-prospective/*`.

## Amendment 1 (2026-10-08): repaired CI outcome model

`ci-prospective-amend1/` adds arms to the prospective CI test without changing the original registration. After the first freeze,
the outcome model was found to treat retriggers of the same failure as independent and to give every test in a suite one
flakiness rate. The repaired DP (per-test flakiness, correlated retriggers, a P(regression) model using the test's label history
and push signals, and a recalibrated belief) was developed on the train window only and scored twice on the retrospective test
window. It wins there with simulated outcomes but loses to "retrigger once, then escalate" on the 169 failures whose
outcomes were all observed. The prospective window, crawled only after this commit, decides. See
`ci-prospective-amend1/PREREG_ci_prospective_amend1.json` for arms, scoring modes, predictions, forecasts and file hashes.

## Telecom v5 (2026-10-08): blind evaluation of the full executor

`telecom-v5/` freezes the tau2-telecom controller v5c (the table/belief picks every move; the controller executes lookups, blocker
transfer, bill payment, refuel, roaming and every phone step; Qwen writes only uncovered turns). Files were hashed at
2026-10-08T13:22:47Z; the evaluation sets were then drawn with the NIST beacon pulse 1975636 (114 fresh reserve tasks, 60 holdout
tasks whose fault never appears in fitting data). Arms, primary comparison and the stated target are in
`telecom-v5/PREREG_telecom_v5.json`. Committed before any evaluation episode finished.
