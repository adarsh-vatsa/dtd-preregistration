"""Run tau2 with the DtD agent registered as 'dtd_agent' (tau2's own code is not modified).
usage: DTD_MODE=explore python dtd_run.py run --domain telecom --agent dtd_agent ... (same flags as `tau2 run`)"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau2.registry import registry
from dtd_telecom_agent import create_dtd_agent
from dtd_retail_agent import create_dtd_retail_agent

registry.register_agent_factory(create_dtd_agent, "dtd_agent")
registry.register_agent_factory(create_dtd_retail_agent, "dtd_retail_agent")

# The natural-language-assertion grader defaults to an OpenAI model (gpt-4.1); we have no OpenAI key, so DTD_NL_JUDGE points it at
# a local model (same judge for every arm; a documented deviation from the official harness).
if os.environ.get("DTD_NL_JUDGE"):
    import json as _json
    import tau2.evaluator.evaluator_nl_assertions as _nl
    _nl.DEFAULT_LLM_NL_ASSERTIONS = os.environ["DTD_NL_JUDGE"]
    _nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = _json.loads(os.environ.get("DTD_NL_JUDGE_ARGS", '{"temperature": 0.0}'))
from tau2.cli import main

if __name__ == "__main__":
    main()
