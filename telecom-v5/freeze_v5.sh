#!/usr/bin/env bash
# Freeze telecom v5 (hash the controller files), take a NIST beacon pulse AFTER the freeze, draw the evaluation sets, write the
# registration, and drop the evaluation script into the GPU work queue. Run on the Jarvis login node:
#   bash ~/dtd-bounded/bounded/freeze_v5.sh
set -euo pipefail
cd ~/dtd-bounded/bounded
sha256sum telecom_tracker.py telecom_v5.py dtd_telecom_agent.py dtd_run.py order_dp.json telecom_splits.json > v5_frozen_sha256.txt
date -u +%Y-%m-%dT%H:%M:%SZ >> v5_frozen_sha256.txt
curl -s --max-time 30 https://beacon.nist.gov/beacon/2.0/pulse/last > v5_beacon_pulse.json
python3 - <<'EOF'
import json, random, datetime, hashlib
sp = json.load(open("telecom_splits.json")); pulse = json.load(open("v5_beacon_pulse.json"))["pulse"]
seed = int(pulse["outputValue"][:16], 16); rng = random.Random(seed)
used = set(x.strip() for x in open("reserve_eval_ids.txt") if x.strip())
res = sorted(t for t in sp["reserve"] if t not in used)
mob = [t for t in res if t.startswith("[mobile_data_issue]")]; mms = [t for t in res if t.startswith("[mms_issue]")]
pick = mob + rng.sample(mms, 114 - len(mob))                      # all remaining mobile-data tasks + MMS to 114
ood = rng.sample(sorted(sp["holdout"]), 60)                        # holdout: every task has the fault never seen in fitting
open("v5_eval_reserve_ids.txt", "w").write("\n".join(pick) + "\n"); open("v5_eval_holdout_ids.txt", "w").write("\n".join(ood) + "\n")
reg = {"written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "frozen_files_sha256": open("v5_frozen_sha256.txt").read(),
       "beacon": {"uri": pulse.get("uri"), "timeStamp": pulse.get("timeStamp"), "outputValue": pulse["outputValue"]},
       "sets": {"reserve (in distribution)": {"n": len(pick), "rule": "all 34 remaining reserve mobile-data tasks + MMS sampled with random.Random(int(outputValue[:16],16))"},
                "holdout (out of distribution: fault break_app_storage_permission never in fitting data)": {"n": len(ood), "rule": "same RNG, after the reserve draw"}},
       "arms": ["Opus 5.5 alone (tau2 llm_agent via claude -p shim; reserve set only, if the shim is reachable)", "Opus 5.5 + DtD v5 (mdp; reserve only)",
                "Qwen free (tau2 llm_agent behaviour inside dtd_agent, mode free)", "Qwen + DtD v5, decide = mdp (static DP order)",
                "Qwen + DtD v5, decide = belief (QMDP over the fault posterior)", "Qwen + DtD v5, decide = pomdp (belief + 2-step lookahead)"],
       "customer": "Qwen3.8-27B, temperature 0, as in every earlier telecom run", "trials": 1,
       "primary": "success of Qwen + DtD v5 (mdp) vs Qwen free on the reserve set (paired by task; McNemar; task-clustered bootstrap)",
       "secondary": ["each v5 decide step vs free", "v5 vs the earlier Opus-alone rate on reserve tasks (111/114 on the first reserve draw; reference only)",
                     "agent LLM calls and billed tokens per episode", "holdout success (OOD)"],
       "target_stated_in_advance": "reach Opus alone's level (about 97% on telecom in this harness) with Qwen + v5; if not reached, report the gap plainly"}
json.dump(reg, open("PREREG_telecom_v5.json", "w"), indent=1)
print(json.dumps({k: reg[k] for k in ("written_utc", "beacon", "sets")}, indent=1))
EOF
cat > ~/dtd-bounded/queue/q_v5_eval.sh <<'EOS'
#!/usr/bin/env bash
cd ~/dtd-bounded && . venv-tau2/bin/activate && cd tau2-bench
export OPENAI_API_KEY=dummy HOSTED_VLLM_API_KEY=dummy DTD_MAX_BLOCKS=3
export DTD_ORDER=/home/avatsa/dtd-bounded/bounded/order_dp.json DTD_TASKS=/home/avatsa/dtd-bounded/tau2-bench/data/tau2/domains/telecom/tasks.json DTD_SPLITS=/home/avatsa/dtd-bounded/bounded/telecom_splits.json
NT="{\"temperature\":0.0,\"api_base\":\"http://localhost:8000/v1\",\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
R=~/dtd-bounded/runs
for SET in reserve holdout; do
  mapfile -t IDS < ~/dtd-bounded/bounded/v5_eval_${SET}_ids.txt
  arm() { mode=$1; dec=$2; out=$3; DTD_MODE=$mode DTD_DECIDE=$dec python ../bounded/dtd_run.py run --domain telecom --task-split-name full --task-ids "${IDS[@]}" --num-trials 1 --agent dtd_agent --agent-llm hosted_vllm/qwen --agent-llm-args "$NT" --user-llm hosted_vllm/qwen --user-llm-args "$NT" --max-concurrency 40 --save-to $R/$out --auto-resume --log-level ERROR > $R/$out.log 2>&1; }
  arm free mdp v5eval_${SET}_qwen_free &
  arm dstx mdp v5eval_${SET}_qwen_mdp &
  arm dstx belief v5eval_${SET}_qwen_belief &
  arm dstx pomdp v5eval_${SET}_qwen_pomdp &
  if [ "$SET" = reserve ] && curl -s -m 10 localhost:18091/v1/models >/dev/null; then   # Opus via the claude -p shim (reverse tunnel)
    OA="{\"api_base\":\"http://localhost:18091/v1\"}"
    ( tau2 run --domain telecom --task-split-name full --task-ids "${IDS[@]}" --num-trials 1 --agent llm_agent --agent-llm openai/opus --agent-llm-args "$OA" --user-llm hosted_vllm/qwen --user-llm-args "$NT" --max-concurrency 4 --save-to $R/v5eval_reserve_opus_free --auto-resume --log-level ERROR > $R/v5eval_reserve_opus_free.log 2>&1 ) &
    ( DTD_MODE=dstx DTD_DECIDE=mdp python ../bounded/dtd_run.py run --domain telecom --task-split-name full --task-ids "${IDS[@]}" --num-trials 1 --agent dtd_agent --agent-llm openai/opus --agent-llm-args "$OA" --user-llm hosted_vllm/qwen --user-llm-args "$NT" --max-concurrency 4 --save-to $R/v5eval_reserve_opus_mdp --auto-resume --log-level ERROR > $R/v5eval_reserve_opus_mdp.log 2>&1 ) &
  fi
  wait
done
echo V5_EVAL_DONE
EOS
sha256sum PREREG_telecom_v5.json
echo QUEUED
