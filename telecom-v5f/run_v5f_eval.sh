#!/usr/bin/env bash
# Registered v5f evaluation on draw 3 (ids in v5f_eval_ids.txt, drawn after the freeze). Qwen arms on the GPU node that runs
# this script (vLLM at localhost:8000); Opus arms through the claude -p shim (localhost:18091). Customer: Qwen, temperature 0.
cd ~/dtd-bounded && . venv-tau2/bin/activate && cd tau2-bench
F=/home/avatsa/dtd-bounded/bounded_v5f_frozen
export OPENAI_API_KEY=dummy HOSTED_VLLM_API_KEY=dummy DTD_MAX_BLOCKS=3 DTD_ORDER=$F/order_dp.json DTD_SPLITS=$F/telecom_splits.json
export DTD_TASKS=/home/avatsa/dtd-bounded/tau2-bench/data/tau2/domains/telecom/tasks.json
NT="{\"temperature\":0.0,\"api_base\":\"http://localhost:8000/v1\",\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
TH="{\"temperature\":0.6,\"top_p\":0.95,\"api_base\":\"http://localhost:8000/v1\",\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":true}}}"
OA="{\"api_base\":\"http://localhost:18091/v1\"}"
R=~/dtd-bounded/runs; mapfile -t IDS < $F/v5f_eval_ids.txt
arm() { llm=$1; largs=$2; mode=$3; dec=$4; chk=$5; out=$6; c=$7; DTD_MODE=$mode DTD_DECIDE=$dec DTD_CHECKLIST=$chk python $F/dtd_run.py run --domain telecom --task-split-name full --task-ids "${IDS[@]}" --num-trials 1 --agent dtd_agent --agent-llm $llm --agent-llm-args "$largs" --user-llm hosted_vllm/qwen --user-llm-args "$NT" --max-concurrency $c --save-to $R/$out --auto-resume --log-level ERROR > $R/$out.log 2>&1; echo "$out DONE" >> $R/$out.log; }
case "${1:-all}" in
 qwen)
  arm hosted_vllm/qwen "$NT" free mdp 0 v5f_eval_qwen_free 16 &
  arm hosted_vllm/qwen "$NT" dstx mdp 0 v5f_eval_qwen_mdp 16 &
  arm hosted_vllm/qwen "$NT" dstx belief 0 v5f_eval_qwen_belief 16 &
  arm hosted_vllm/qwen "$NT" dstx pomdp 0 v5f_eval_qwen_pomdp 16 &
  arm hosted_vllm/qwen "$TH" dstx mdp 0 v5f_eval_qwen_thinking_mdp 12 &
  arm hosted_vllm/qwen "$TH" free mdp 1 v5f_eval_qwen_thinking_checklist 8 & ;;
 opus)
  ( tau2 run --domain telecom --task-split-name full --task-ids "${IDS[@]}" --num-trials 1 --agent llm_agent --agent-llm openai/opus --agent-llm-args "$OA" --user-llm hosted_vllm/qwen --user-llm-args "$NT" --max-concurrency 4 --save-to $R/v5f_eval_opus_free --auto-resume --log-level ERROR > $R/v5f_eval_opus_free.log 2>&1; echo "v5f_eval_opus_free DONE" >> $R/v5f_eval_opus_free.log ) &
  arm openai/opus "$OA" dstx mdp 0 v5f_eval_opus_mdp 4 & ;;
esac
wait; echo "V5F_EVAL_${1:-all}_DONE"
