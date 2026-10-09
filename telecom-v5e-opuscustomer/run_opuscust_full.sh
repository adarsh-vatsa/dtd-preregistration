#!/usr/bin/env bash
# Registered robustness test: Claude Opus 5.5 as the simulated customer on the registered v5e draw.
# Phase 1 (confirmatory): Qwen alone and Qwen + v5e on the 64 draw tasks never run with an Opus customer.
# Phase 2 (secondary): Qwen with thinking + checklist and Opus + v5e on all 114 draw tasks.
# Phase 3 (secondary): Opus alone on all 114 draw tasks; Qwen with thinking alone on all 114.
# Controller code: ~/dtd-bounded/bounded_v5e_ablate (executor, tracker and runner byte-identical to the registered v5e).
cd ~/dtd-bounded && . venv-tau2/bin/activate && cd tau2-bench
F=/home/avatsa/dtd-bounded/bounded_v5e_ablate
export OPENAI_API_KEY=dummy HOSTED_VLLM_API_KEY=dummy DTD_MAX_BLOCKS=3 DTD_ORDER=$F/order_dp.json DTD_SPLITS=$F/telecom_splits.json
export DTD_TASKS=/home/avatsa/dtd-bounded/tau2-bench/data/tau2/domains/telecom/tasks.json
NT="{\"temperature\":0.0,\"api_base\":\"http://localhost:8000/v1\",\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
TH="{\"temperature\":0.6,\"top_p\":0.95,\"api_base\":\"http://localhost:8000/v1\",\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":true}}}"
OA="{\"api_base\":\"http://localhost:18091/v1\"}"
R=~/dtd-bounded/runs
mapfile -t C64 < <(sed -n 51,114p /home/avatsa/dtd-bounded/bounded_v5e_frozen/v5e_eval_ids.txt)
mapfile -t ALL < /home/avatsa/dtd-bounded/bounded_v5e_frozen/v5e_eval_ids.txt
arm() { agent=$1; largs=$2; mode=$3; chk=$4; out=$5; c=$6; shift 6
  DTD_MODE=$mode DTD_DECIDE=mdp DTD_NO_LLM=0 DTD_CHECKLIST=$chk python $F/dtd_run.py run --domain telecom --task-split-name full --task-ids "$@" --num-trials 1 --agent dtd_agent --agent-llm $agent --agent-llm-args "$largs" --user-llm openai/opus --user-llm-args "$OA" --max-concurrency $c --save-to $R/$out --auto-resume --log-level ERROR > $R/$out.log 2>&1; echo "$out DONE" >> $R/$out.log; }
arm hosted_vllm/qwen "$NT" free 0 oc_v5e_c64_qwen_free 4 "${C64[@]}" &
arm hosted_vllm/qwen "$NT" dstx 0 oc_v5e_c64_qwen_mdp 4 "${C64[@]}" &
wait; echo PHASE1_DONE
arm hosted_vllm/qwen "$TH" free 1 oc_v5e_all_qwen_thinking_checklist 4 "${ALL[@]}" &
arm openai/opus "$OA" dstx 0 oc_v5e_all_opus_mdp 4 "${ALL[@]}" &
wait; echo PHASE2_DONE
arm openai/opus "$OA" free 0 oc_v5e_all_opus_free 5 "${ALL[@]}" &
arm hosted_vllm/qwen "$TH" free 0 oc_v5e_all_qwen_thinking 3 "${ALL[@]}" &
wait; echo OPUSCUST_DONE
