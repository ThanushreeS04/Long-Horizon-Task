import copy
from tests.test_workflow import make_task
from agents.llm import MockLLM
from recovery_bench import FaultyLLM
from agents.workflow import WorkflowBaselineAgent

t = make_task(8)
llm = FaultyLLM(MockLLM(), {2: 99})
r = WorkflowBaselineAgent(llm, replan=True).run(copy.deepcopy(t))
print("completed", r["completed"], "replans", r["replans"], "blocked", r["blocked"])
print(r["executed"])