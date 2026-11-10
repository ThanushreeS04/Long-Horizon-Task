"""Simple dashboard for the Long-Horizon Agent project.

Run from the long_horizon_agent project folder:
    python frontend.py
Open http://127.0.0.1:8000
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import json
import os
import re
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parent
SOLVE = ROOT / "solve.py"
DATA = ROOT / "data"
RUN_LOCK = threading.Lock()
CHARTS = DATA / "results" / "charts"
CURRENT = {"proc": None, "stopped": False}      # the running solve.py, so Stop can end it
AGENTS = {"all": "all", "baseline+memory": "both", "memory": "memory", "mea": "mea", "baseline": "baseline"}

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Long-Horizon Agent | Insight Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root{--bg:#f7f7fc;--panel:#fff;--ink:#171936;--muted:#70738f;--line:#e8e7f3;--purple:#7254df;--purple2:#eeeaff;--blue:#4c91f0;--green:#39b98b;--orange:#efa34b;--shadow:0 8px 28px #34236c0b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px 'DM Sans',sans-serif}
.wrap{max-width:1220px;margin:auto;padding:34px 30px 50px}.header{display:flex;justify-content:space-between;align-items:center;margin-bottom:26px}.brand{font-size:14px;font-weight:800;letter-spacing:.2px}.brand span{color:var(--purple)}.local{font-size:12px;color:#328b6e;background:#e8f8f1;border:1px solid #ccefe1;padding:8px 12px;border-radius:20px}
h1{font-size:30px;letter-spacing:-1px;margin:0 0 6px;font-weight:800}.sub{margin:0;color:var(--muted);font-size:14px}.section{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:22px 24px;box-shadow:var(--shadow);margin-top:22px}.section-head{font-size:16px;font-weight:800;margin:0 0 16px}.section-head small{display:block;color:var(--muted);font-weight:400;font-size:12px;margin-top:3px}
label{display:block;font-size:12px;font-weight:700;color:#4e506d;margin:0 0 8px}textarea,input,select{font:inherit;width:100%;border:1px solid #dedced;border-radius:11px;background:#fff;color:var(--ink);padding:12px 13px;outline:none}textarea{min-height:104px;resize:vertical;line-height:1.5}textarea:focus,input:focus,select:focus{border-color:#9a86f0;box-shadow:0 0 0 3px #7254df14}.settings{display:grid;grid-template-columns:1fr 1fr 1fr 1.25fr auto;gap:14px;align-items:end;margin-top:16px}.run{border:0;border-radius:11px;background:var(--purple);color:#fff;font-weight:700;padding:13px 20px;cursor:pointer;white-space:nowrap;height:43px}.run:hover{background:#5e42cb}.run:disabled{opacity:.55;cursor:wait}.model-wrap{display:none;margin-top:14px;max-width:320px}.status{display:none;margin-top:14px;padding:10px 12px;border-radius:10px;background:#f1efff;color:#5742b4;font-size:12px}.status.error{background:#fff0f0;color:#b43f4a}
.results{display:none}.results-head{display:flex;justify-content:space-between;align-items:end;gap:15px;margin-bottom:15px}.results-head h2{font-size:20px;margin:0}.meta{font-size:11px;color:var(--muted)}.success{font-size:12px;color:#23835f;background:#e9f8f1;border:1px solid #ccefe1;padding:7px 11px;border-radius:20px;white-space:nowrap}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{border:1px solid var(--line);border-radius:14px;padding:16px;background:#fff;min-height:104px}.metric:nth-child(1){background:#f4f0ff}.metric:nth-child(2){background:#eef5ff}.metric:nth-child(3){background:#edfaf5}.metric:nth-child(4){background:#fff6ec}.metric-label{font-size:12px;color:#626580}.metric-value{font-size:25px;font-weight:800;letter-spacing:-.7px;margin-top:8px}.metric-note{font-size:11px;color:var(--muted);margin-top:2px}
.lower{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(260px,.8fr);gap:14px;margin-top:14px}.subpanel{border:1px solid var(--line);border-radius:14px;padding:17px;background:#fff;min-width:0}.subpanel h3{font-size:14px;margin:0 0 16px}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-bottom:16px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.bar-group{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.bar-card{background:#fafaff;border:1px solid #efedf8;border-radius:11px;padding:12px}.bar-title{font-size:11px;color:var(--muted);margin-bottom:12px}.bars{height:112px;display:flex;align-items:end;gap:8px;border-bottom:1px solid #dcd9ec;padding:0 5px}.bar-col{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:end;height:100%;gap:5px}.bar{width:100%;max-width:38px;border-radius:6px 6px 0 0;min-height:3px}.bar-value{font-size:10px;color:#555873}.bar-name{font-size:10px;color:var(--muted);margin-top:7px}.report-status{display:inline-block;color:#21845e;background:#e9f8f1;padding:6px 10px;border-radius:20px;font-size:11px;font-weight:700;margin-bottom:12px}.report-text{font-size:12px;color:#565a76;line-height:1.6;white-space:pre-wrap;max-height:230px;overflow:auto}.table-wrap{overflow:auto;margin-top:14px;border:1px solid var(--line);border-radius:12px}table{width:100%;border-collapse:collapse;min-width:600px}th,td{text-align:left;padding:11px 13px;border-bottom:1px solid var(--line);font-size:11px}th{background:#f8f7fd;color:#666983;font-size:10px;text-transform:uppercase;letter-spacing:.5px}td:first-child{font-weight:700;color:#4d506c}tr:last-child td{border-bottom:0}.details{margin-top:12px;border:1px solid var(--line);border-radius:11px;background:#fff}.details summary{cursor:pointer;padding:12px 14px;font-size:12px;font-weight:700}.details pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto;padding:13px;margin:0;border-top:1px solid var(--line);font:11px/1.5 Consolas,monospace;color:#4c5069;background:#fafaff}
footer{margin-top:24px;color:#9a9bb0;font-size:10px;text-align:center}
@media(max-width:850px){.settings{grid-template-columns:1fr 1fr}.lower{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){.wrap{padding:22px 14px 35px}.header{margin-bottom:22px}h1{font-size:25px}.section{padding:17px}.settings{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}.results-head{align-items:start;flex-direction:column}.bar-group{grid-template-columns:1fr}}
</style></head>
<body><main class="wrap">
<header class="header"><div class="brand"><span>HORIZON</span> &nbsp;|&nbsp; AGENT INSIGHTS</div><div class="local">● Local project</div></header>
<div><h1>Agent Insight Dashboard</h1><p class="sub">Run one task and compare how Baseline, Memory and MEA perform.</p></div>
<section class="section">
 <h2 class="section-head">Run an experiment<small>Enter your task and choose the execution settings.</small></h2>
 <form id="taskForm"><label for="task">Task description</label><textarea id="task" required placeholder="e.g. Plan and design a student attendance management system. Include requirements, database design, implementation steps and testing."></textarea>
 <div class="settings"><div><label for="steps">Number of steps</label><input id="steps" type="number" min="3" max="30" value="8"></div><div><label for="backend">Backend</label><select id="backend"><option value="mock">Mock</option><option value="ollama">Ollama</option></select></div><div><label for="model">Model name</label><input id="model" value="qwen2.5:3b"></div><div><label for="agents">Agents to run</label><select id="agents"><option value="all">All three</option><option value="baseline+memory">Baseline + Memory</option><option value="memory">Memory only</option><option value="mea">MEA only</option><option value="baseline">Baseline only</option></select></div><button class="run" id="runBtn" type="submit">Run comparison →</button></div>
 <div class="status" id="status"></div><button class="run" id="stopBtn" type="button" style="display:none;margin-top:12px;background:#b43f4a">Stop run</button></form>
</section>
<section class="section results" id="results">
 <div class="results-head"><div><h2>Results overview</h2><div class="meta" id="runMeta"></div></div><div class="success" id="runStatus">● Run complete</div></div>
 <div class="metrics" id="metrics"></div>
 <div class="lower"><div class="subpanel"><h3>Agent comparison</h3><div class="legend"><span><i class="dot" style="background:#4c91f0"></i>Baseline</span><span><i class="dot" style="background:#39b98b"></i>Memory</span><span><i class="dot" style="background:#a184ed"></i>MEA</span></div><div class="bar-group" id="charts"></div></div>
 <div class="subpanel"><h3>Final report</h3><div class="report-status" id="reportStatus">Report generated</div><div class="report-text" id="finalOutput"></div></div></div>
 <div class="table-wrap"><table><thead id="tableHead"></thead><tbody id="tableBody"></tbody></table></div>
 <details class="details"><summary>Cost by role (MEA)</summary><div class="table-wrap" style="margin:0;border:0"><table><thead><tr><th>Role</th><th>Calls</th><th>In tokens</th><th>Out tokens</th><th>% of tokens</th><th>Seconds</th><th>Model</th></tr></thead><tbody id="roleBody"></tbody></table></div></details><details class="details"><summary>Step execution details</summary><pre id="stepOutput"></pre></details><details class="details"><summary>Raw experiment data</summary><pre id="rawOutput"></pre></details>
</section>
<section class="section" id="saved"><h2 class="section-head">Saved benchmark charts<small>Read-only. Drawn by make_charts.py from data/results; each title says whether it is a mock or real-model run.</small></h2><div id="savedCharts">Loading…</div></section><footer>Long-Horizon Agent · Memory Management Module</footer></main>
<script>
const $=id=>document.getElementById(id),backend=$('backend');
backend.addEventListener('change',()=>{$('model').disabled=backend.value!=='ollama';$('model').style.opacity=backend.value==='ollama'?'1':'.55'});$('model').disabled=true;$('model').style.opacity='.55';
const fmt=n=>typeof n==='number'?n.toLocaleString(undefined,{maximumFractionDigits:2}):(n??'—');
const get=(o,k,d=0)=>o&&o[k]!==undefined?o[k]:d;
function metric(label,value,note){return `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-note">${note||''}</div></div>`}
function chart(title,key,agents,lowerBetter=false){const vals=agents.map(a=>Number(get(a[1],key,0))||0),max=Math.max(...vals,1);return `<div class="bar-card"><div class="bar-title">${title}</div><div class="bars">${agents.map((a,i)=>{let h=vals[i]?Math.max(4,vals[i]/max*88):3;return `<div class="bar-col"><div class="bar-value">${fmt(vals[i])}</div><div class="bar" style="height:${h}px;background:${({Baseline:'#4c91f0',Memory:'#39b98b',MEA:'#a184ed'})[agents[i][0]]}"></div></div>`}).join('')}</div><div style="display:flex;justify-content:space-around">${agents.map(a=>`<span class="bar-name">${a[0]}</span>`).join('')}</div></div>`}
$('taskForm').addEventListener('submit',async e=>{e.preventDefault();const btn=$('runBtn'),status=$('status');btn.disabled=true;btn.textContent='Running…';$('stopBtn').style.display='inline-block';status.style.display='block';status.className='status';status.textContent='Running the selected agents. Ollama may take a few minutes; press Stop to cancel.';$('results').style.display='none';try{const res=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:$('task').value,steps:Number($('steps').value),backend:backend.value,agents:$('agents').value,model:$('model').value.trim()})});const data=await res.json();if(!res.ok)throw Error(data.error||'Run failed');render(data);status.textContent='Experiment completed. Results are shown below.';$('results').style.display='block';$('results').scrollIntoView({behavior:'smooth'});}catch(err){status.className='status error';status.textContent=err.message}finally{btn.disabled=false;btn.textContent='Run comparison →';$('stopBtn').style.display='none'}});
$('stopBtn').addEventListener('click',async()=>{try{await fetch('/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})}catch(e){}});
fetch('/charts-list').then(r=>r.json()).then(list=>{const box=$('savedCharts');box.textContent='';if(!list.length){box.textContent='No charts yet. Run: python make_charts.py';return}list.forEach(n=>{const im=document.createElement('img');im.src='/charts/'+encodeURIComponent(n);im.alt=n;im.style.cssText='width:100%;max-width:760px;display:block;margin:10px 0;border:1px solid #e8e7f3;border-radius:12px';box.appendChild(im)})}).catch(()=>{$('savedCharts').textContent='Charts unavailable.'});
function render(data){const r=data.results||{},agents=[['Baseline',r.baseline],['Memory',r.memory],['MEA',r.mea]].filter(a=>a[1]);const preferred=r.memory||r.mea||r.baseline||{};$('runMeta').textContent=`${data.backend||'mock'}${data.model?' · '+data.model:''} · ${data.steps} steps`;$('runStatus').textContent=preferred.completed?'✓ Completed':'● Run finished';$('metrics').innerHTML=[metric('Completion rate',preferred.completed?'100%':'Incomplete','Selected: '+(r.memory?'Memory':r.mea?'MEA':'Baseline')),metric('Verified steps',`${fmt(get(preferred,'verified_steps'))} / ${data.steps}`,'Steps verified'),metric('Input tokens',fmt(get(preferred,'input_tokens')),'Selected agent'),metric('Execution time',fmt(get(preferred,'elapsed_seconds'))+'s','Selected agent')].join('');
$('charts').innerHTML=chart('Completion rate (%)','completed',agents.map(a=>[a[0],{completed:get(a[1],'completed')?100:0}]))+chart('Input tokens','input_tokens',agents)+chart('Execution time (s)','elapsed_seconds',agents);
const rows=[['Completed',x=>x.completed?'Yes':'No'],['Verified steps',x=>fmt(get(x,'verified_steps'))],['LLM calls',x=>fmt(get(x,'llm_calls'))],['Rejected attempts',x=>fmt(get(x,'rejected_attempts'))],['Replans',x=>fmt(get(x,'replans'))],['Repeated steps after crash',x=>fmt(get(x,'repeated_steps'))],['Input tokens',x=>fmt(get(x,'input_tokens'))],['Output tokens',x=>fmt(get(x,'output_tokens'))],['Peak prompt tokens',x=>fmt(get(x,'peak_prompt_tokens'))],['Facts recalled',x=>`${fmt(get(x,'facts_recalled'))} / ${fmt(get(x,'facts_total'))}`],['Memory overhead',x=>fmt(get(x,'memory_seconds'))+'s'],['Execution time',x=>fmt(get(x,'elapsed_seconds'))+'s']];
$('tableHead').innerHTML='<tr><th>Metric</th>'+agents.map(a=>`<th>${a[0]}</th>`).join('')+'</tr>';$('tableBody').innerHTML=rows.map(([label,fn])=>`<tr><td>${label}</td>${agents.map(a=>`<td>${fn(a[1])}</td>`).join('')}</tr>`).join('');$('finalOutput').textContent=preferred.final_output||'No final report was returned.';const steps=preferred.step_outputs||{};$('stepOutput').textContent=JSON.stringify(steps,null,2)||'No step details returned.';$('rawOutput').textContent=JSON.stringify(data,null,2);$('reportStatus').textContent=preferred.completed?'✓ Requirements verified':'Review execution output';
const led=(r.mea||{}).ledger;$('roleBody').innerHTML=led?Object.entries(led).map(([role,x])=>`<tr><td>${role}</td><td>${fmt(x.calls)}</td><td>${fmt(x.input_tokens)}</td><td>${fmt(x.output_tokens)}</td><td>${(100*(x.token_share||0)).toFixed(1)}%</td><td>${fmt(x.seconds)}</td><td>${x.model||'rule-based'}</td></tr>`).join(''):'<tr><td colspan="7">Run MEA to see its cost by role.</td></tr>';}
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def send_data(self, status, payload, content_type="application/json; charset=utf-8"):
        body = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path in ("/", "/index.html"):
            return self.send_data(200, PAGE, "text/html; charset=utf-8")
        path = urlparse(self.path).path
        if path == "/charts-list":
            return self.send_data(200, sorted(p.name for p in CHARTS.glob("*.svg")) if CHARTS.is_dir() else [])
        m = re.fullmatch(r"/charts/([A-Za-z0-9_.-]+\.svg)", path)
        if m and (CHARTS / m.group(1)).is_file():
            return self.send_data(200, (CHARTS / m.group(1)).read_text(encoding="utf-8"), "image/svg+xml")
        self.send_data(404, {"error": "Not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try: length = max(0, int(self.headers.get("Content-Length") or 0))
        except ValueError: length = 0
        raw_body = self.rfile.read(min(length, 65536))      # read first: replying with an unread body can reset the connection
        if path not in ("/run", "/stop"): return self.send_data(404, {"error": "Not found"})
        # Only this page may start a run. A request from another website (CSRF) or through a
        # rebound DNS name is refused; JSON is required so browsers must send a preflight first.
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if (self.headers.get("Host") or "").lower() not in hosts:
            return self.send_data(403, {"error": "Forbidden host."})
        origin = self.headers.get("Origin")
        if origin and origin.lower() not in {f"http://{h}" for h in hosts}:
            return self.send_data(403, {"error": "Forbidden origin."})
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self.send_data(415, {"error": "Send JSON."})
        if path == "/stop":
            p = CURRENT["proc"]
            if p is None or p.poll() is not None: return self.send_data(200, {"stopped": False})
            CURRENT["stopped"] = True
            p.kill()
            return self.send_data(200, {"stopped": True})
        if not SOLVE.exists(): return self.send_data(500, {"error": "Place frontend.py beside solve.py in your project root."})
        try:
            req = json.loads(raw_body)
            task = str(req.get("task", "")).strip(); steps = int(req.get("steps", 8))
            backend = req.get("backend", "mock"); model = str(req.get("model", "qwen2.5:3b")).strip()
            if not task: return self.send_data(400, {"error": "Please enter a task."})
            if task.startswith("-") or len(task) > 500:
                return self.send_data(400, {"error": "Task must be under 500 characters and not start with '-'."})
            if not 3 <= steps <= 30: return self.send_data(400, {"error": "Choose between 3 and 30 steps."})
            if backend not in ("mock", "ollama"): return self.send_data(400, {"error": "Unsupported backend."})
            agents = str(req.get("agents", "all"))
            if agents not in AGENTS: return self.send_data(400, {"error": "Unsupported agent choice."})
            if backend == "ollama" and not model: return self.send_data(400, {"error": "Enter an Ollama model name."})
            if not RUN_LOCK.acquire(blocking=False): return self.send_data(409, {"error": "A task is already running. Please wait."})
            try:
                DATA.mkdir(exist_ok=True)
                before = set(DATA.glob("*_comparison.json"))
                cmd = [sys.executable, str(SOLVE), task, "--steps", str(steps), "--only", AGENTS[agents], "--llm", backend]
                if backend == "ollama": cmd += ["--model", model]
                env = dict(os.environ)
                env.setdefault("LH_TIMEOUT", "300")        # one stuck model call cannot block the run for 15 minutes
                CURRENT["stopped"] = False
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                        encoding="utf-8", errors="replace", env=env)
                CURRENT["proc"] = proc
                try:
                    out, err = proc.communicate(timeout=900)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.communicate()
                    raise
                finally:
                    CURRENT["proc"] = None
                if CURRENT["stopped"]: return self.send_data(400, {"error": "Run stopped."})
                if proc.returncode != 0:
                    return self.send_data(500, {"error": (err or out or "Agent execution failed")[-5000:]})
                candidates = [p for p in DATA.glob("*_comparison.json") if p not in before]
                if not candidates: return self.send_data(500, {"error": "Run finished, but the comparison JSON was not found."})
                result_path = max(candidates, key=lambda p: p.stat().st_mtime)
                raw = json.loads(result_path.read_text(encoding="utf-8"))
                payload = {"task": raw.get("task", {}).get("goal", task), "steps": len(raw.get("task", {}).get("steps", [])),
                           "backend": raw.get("backend", backend), "model": raw.get("model"),
                           "results": {k: raw[k] for k in ("baseline", "memory", "mea") if k in raw},
                           "result_file": str(result_path.relative_to(ROOT))}
                return self.send_data(200, payload)
            finally: RUN_LOCK.release()
        except subprocess.TimeoutExpired:
            return self.send_data(504, {"error": "The run exceeded 15 minutes. Try fewer steps or Mock."})
        except Exception as exc:
            return self.send_data(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt, *args): print("[frontend] " + fmt % args)

if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Dashboard running at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop.")
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nStopping dashboard."); server.server_close()
