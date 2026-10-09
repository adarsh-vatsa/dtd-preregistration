#!/usr/bin/env bash
# Registered test: frozen v5f on 60 beacon-drawn holdout tasks (every task contains the fault type absent from all fitting data),
# Claude Opus 5.5 as the simulated customer, 3 trials per task. Qwen arms use vLLM on this node; Opus through the claude -p shim.
cd ~/dtd-bounded && . venv-tau2/bin/activate && cd tau2-bench
F=/home/avatsa/dtd-bounded/bounded_v5f_frozen; H=$F/holdout_oc
export OPENAI_API_KEY=dummy HOSTED_VLLM_API_KEY=dummy DTD_MAX_BLOCKS=3 DTD_ORDER=$F/order_dp.json DTD_SPLITS=$F/telecom_splits.json
export DTD_TASKS=/home/avatsa/dtd-bounded/tau2-bench/data/tau2/domains/telecom/tasks.json
NT="{\"temperature\":0.0,\"api_base\":\"http://localhost:8000/v1\",\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
TH="{\"temperature\":0.6,\"top_p\":0.95,\"api_base\":\"http://localhost:8000/v1\",\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":true}}}"
OA="{\"api_base\":\"http://localhost:18091/v1\"}"
R=~/dtd-bounded/runs; mapfile -t IDS < $H/holdout_oc_ids.txt
arm() { llm=$1; largs=$2; mode=$3; out=$4; c=$5; DTD_MODE=$mode DTD_DECIDE=mdp DTD_CHECKLIST=0 python $F/dtd_run.py run --domain telecom --task-split-name full --task-ids "${IDS[@]}" --num-trials 3 --agent dtd_agent --agent-llm $llm --agent-llm-args "$largs" --user-llm openai/opus --user-llm-args "$OA" --max-concurrency $c --save-to $R/$out --auto-resume --log-level ERROR > $R/$out.log 2>&1; echo "$out DONE" >> $R/$out.log; }
arm hosted_vllm/qwen "$NT" free hoc_v5f_qwen_free 3 &
arm hosted_vllm/qwen "$NT" dstx hoc_v5f_qwen_mdp 3 &
arm hosted_vllm/qwen "$TH" dstx hoc_v5f_qwen_thinking_mdp 3 &
arm openai/opus "$OA" dstx hoc_v5f_opus_mdp 2 &
( tau2 run --domain telecom --task-split-name full --task-ids "${IDS[@]}" --num-trials 3 --agent llm_agent --agent-llm openai/opus --agent-llm-args "$OA" --user-llm openai/opus --user-llm-args "$OA" --max-concurrency 2 --save-to $R/hoc_v5f_opus_free --auto-resume --log-level ERROR > $R/hoc_v5f_opus_free.log 2>&1; echo "hoc_v5f_opus_free DONE" >> $R/hoc_v5f_opus_free.log ) &
wait; echo HOC_DONE
