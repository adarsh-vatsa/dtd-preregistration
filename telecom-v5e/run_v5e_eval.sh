#!/usr/bin/env bash
cd ~/dtd-bounded && . venv-tau2/bin/activate && cd tau2-bench
F=/home/avatsa/dtd-bounded/bounded_v5e_frozen
export OPENAI_API_KEY=dummy HOSTED_VLLM_API_KEY=dummy DTD_MAX_BLOCKS=3 DTD_ORDER=$F/order_dp.json DTD_SPLITS=$F/telecom_splits.json
export DTD_TASKS=/home/avatsa/dtd-bounded/tau2-bench/data/tau2/domains/telecom/tasks.json
NT="{\"temperature\":0.0,\"api_base\":\"http://localhost:8000/v1\",\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
OA="{\"api_base\":\"http://localhost:18091/v1\"}"
R=~/dtd-bounded/runs; mapfile -t IDS < $F/v5e_eval_ids.txt
arm() { llm=$1; largs=$2; mode=$3; dec=$4; out=$5; c=$6; DTD_MODE=$mode DTD_DECIDE=$dec python $F/dtd_run.py run --domain telecom --task-split-name full --task-ids "${IDS[@]}" --num-trials 1 --agent dtd_agent --agent-llm $llm --agent-llm-args "$largs" --user-llm hosted_vllm/qwen --user-llm-args "$NT" --max-concurrency $c --save-to $R/$out --auto-resume --log-level ERROR > $R/$out.log 2>&1; }
arm hosted_vllm/qwen "$NT" free mdp v5e_eval_qwen_free 16 &
arm hosted_vllm/qwen "$NT" dstx mdp v5e_eval_qwen_mdp 16 &
arm hosted_vllm/qwen "$NT" dstx belief v5e_eval_qwen_belief 16 &
arm hosted_vllm/qwen "$NT" dstx pomdp v5e_eval_qwen_pomdp 16 &
( tau2 run --domain telecom --task-split-name full --task-ids "${IDS[@]}" --num-trials 1 --agent llm_agent --agent-llm openai/opus --agent-llm-args "$OA" --user-llm hosted_vllm/qwen --user-llm-args "$NT" --max-concurrency 4 --save-to $R/v5e_eval_opus_free --auto-resume --log-level ERROR > $R/v5e_eval_opus_free.log 2>&1 ) &
arm openai/opus "$OA" dstx mdp v5e_eval_opus_mdp 4 &
wait; echo V5E_EVAL_DONE
