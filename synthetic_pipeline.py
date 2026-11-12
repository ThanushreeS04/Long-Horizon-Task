"""Build grounded Planner and Executor SFT data from recorded trajectories.

Input JSONL, one record per line:
 {"query":"...", "trajectory":[{"observation":"...","action":"..."}], "success":true}
Only recorded action indices may be referenced by generated plans. Failed examples are
excluded unless explicitly requested. Every emitted plan is schema- and grounding-checked.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from agents.llm import make_llm


def parse_json(text):
    cleaned=re.sub(r"^```(?:json)?|```$", "", str(text).strip(), flags=re.I|re.M).strip()
    try: return json.loads(cleaned)
    except json.JSONDecodeError: return None


def valid_steps(steps, n):
    if not isinstance(steps,list) or not steps: return []
    valid=[]
    for st in steps:
        if not isinstance(st,dict): continue
        title=str(st.get("title","")).strip()
        ids=st.get("action_indices",[])
        if title and isinstance(ids,list) and ids and all(type(i) is int and 0<=i<n for i in ids):
            valid.append({"title":title[:240],"action_indices":sorted(set(ids))})
    return valid


def dump_jsonl(path, rows):
    with path.open("w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True); ap.add_argument("--output",default="data/synthetic")
    ap.add_argument("--llm",choices=["mock","ollama","anthropic"],default="ollama")
    ap.add_argument("--model"); ap.add_argument("--expand",type=int,default=2)
    ap.add_argument("--targeted-augment",type=int,default=0,
                    help="Add up to N focused Planner examples per trajectory, grounded to selected action indices")
    ap.add_argument("--include-failures",action="store_true")
    args=ap.parse_args()
    llm=make_llm(args.llm,args.model); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    planner=[]; executor=[]; expanded=[]; targeted=[]; rejected=[]; accepted=0
    source=Path(args.input)
    for line_no,line in enumerate(source.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        try: item=json.loads(line)
        except json.JSONDecodeError as e:
            rejected.append({"line":line_no,"reason":f"invalid JSON: {e}"}); continue
        query=str(item.get("query","")).strip(); traj=item.get("trajectory")
        if not query or not isinstance(traj,list) or not traj:
            rejected.append({"line":line_no,"reason":"missing query or non-empty trajectory"}); continue
        if not item.get("success",False) and not args.include_failures:
            rejected.append({"line":line_no,"reason":"trajectory not marked successful"}); continue
        if any(not isinstance(a,dict) or not str(a.get("action","")).strip() for a in traj):
            rejected.append({"line":line_no,"reason":"trajectory contains malformed action"}); continue
        evidence=json.dumps(traj,ensure_ascii=False,indent=2)
        prompt=("Convert this recorded trajectory into a high-level plan. Every step must cite "
                "the 0-based action indices that ground it. Do not add unrecorded actions. "
                "Return JSON {rationale_summary, steps:[{title,action_indices}]} only.\n"
                f"QUERY: {query}\nTRAJECTORY:\n{evidence}")
        raw=llm.complete(prompt,kind="grounded_plan",max_tokens=1200,
                         meta={"action_count":len(traj)})["text"]
        parsed=parse_json(raw); steps=valid_steps(parsed.get("steps") if isinstance(parsed,dict) else None,len(traj))
        if not steps:
            rejected.append({"line":line_no,"reason":"model produced no valid grounded plan"}); continue
        accepted+=1
        plan_obj={"steps":steps}
        planner.append({"messages":[{"role":"user","content":"Create a high-level plan for: "+query},
                        {"role":"assistant","content":json.dumps(plan_obj,ensure_ascii=False)}]})
        for idx,act in enumerate(traj):
            executor.append({"messages":[{"role":"user","content":
                f"Goal: {query}\nObservation: {act.get('observation','')}\n"
                f"Grounded plan: {json.dumps(steps,ensure_ascii=False)}\n"
                "Return the next action grounded in this observation."},
                {"role":"assistant","content":str(act["action"])}]})
        for k in range(max(0,args.expand)):
            exp=llm.complete("Create a paraphrased task query with the same grounded action plan. "
                "Return JSON {query,steps}; preserve action_indices exactly and do not add actions.\n"
                +json.dumps({"query":query,"steps":steps},ensure_ascii=False),
                kind="plan_expand",max_tokens=800,meta={"query":query,"steps":steps})["text"]
            ex=parse_json(exp)
            ex_steps=valid_steps(ex.get("steps") if isinstance(ex,dict) else None,len(traj))
            if isinstance(ex,dict) and str(ex.get("query","")).strip() and ex_steps:
                expanded.append({"messages":[{"role":"user","content":"Plan this task: "+str(ex["query"])},
                    {"role":"assistant","content":json.dumps({"steps":ex_steps},ensure_ascii=False)}]})
        # Targeted augmentation emphasizes selected action indices while retaining exact grounding.
        declared=item.get("failure_prone_indices", [])
        targets=[i for i in declared if type(i) is int and 0<=i<len(traj)] if isinstance(declared,list) else []
        if not targets: targets=list(range(len(traj)))
        for target in targets[:max(0,args.targeted_augment)]:
            target_prompt=("Create one targeted Planner example focusing on the supplied action index. "
                "Every step's action_indices must be a subset of the target index list. "
                "Do not invent actions. Return JSON {query,steps:[{title,action_indices}]} only.\n"
                +json.dumps({"query":query,"target_indices":[target],"trajectory":traj},ensure_ascii=False))
            aug=llm.complete(target_prompt,kind="targeted_augment",max_tokens=600,
                meta={"query":query,"target_index":target})["text"]
            aug_obj=parse_json(aug)
            aug_steps=valid_steps(aug_obj.get("steps") if isinstance(aug_obj,dict) else None,len(traj))
            if isinstance(aug_obj,dict) and str(aug_obj.get("query","")).strip() and aug_steps and all(
                    set(st["action_indices"]) <= {target} for st in aug_steps):
                targeted.append({"messages":[{"role":"user","content":"Create a focused plan for: "+str(aug_obj["query"])},
                    {"role":"assistant","content":json.dumps({"steps":aug_steps},ensure_ascii=False)}]})
    dump_jsonl(out/"planner_train.jsonl",planner+expanded+targeted)
    dump_jsonl(out/"executor_train.jsonl",executor)
    report={"input":str(source),"accepted_trajectories":accepted,"rejected":rejected,
            "planner_examples":len(planner)+len(expanded)+len(targeted),"grounded_plans":len(planner),
            "expanded_plans":len(expanded),"targeted_augmentations":len(targeted),
            "executor_examples":len(executor)}
    (out/"pipeline_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
