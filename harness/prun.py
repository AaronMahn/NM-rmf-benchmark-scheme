import sys, os, json, random, subprocess, glob, time
ROOT=os.environ.get("P26_ROOT",os.path.expanduser("~/P26exp")); HOME=os.path.expanduser("~")
MODELS={"gpt55":"openrouter/openai/gpt-5.5","claude":"openrouter/anthropic/claude-opus-4.8",
        "gemini":"google/gemini-3.1-pro-preview","qwen":"dashscope/qwen3.7-max",
        "deepseek":"deepseek/deepseek-v4-pro","kimi":"moonshot/kimi-k2.6"}
RUNTAG="prun"; TIMEOUT="600"; NS=3
MAN=f"{ROOT}/results/parallel_manifest.json"
model=sys.argv[1]
man=json.load(open(MAN)) if os.path.exists(MAN) else {}
samples = man[model]["samples"] if model in man else sorted(random.sample(range(100),NS))
env=dict(os.environ); env["PATH"]="/opt/homebrew/bin:"+env.get("PATH","")
subprocess.run(["openclaw","hooks","disable","session-memory"],env=env)
def hygiene(agent):
    sd=f"{HOME}/.openclaw/agents/{agent}/sessions"
    for f in glob.glob(sd+"/*.jsonl")+glob.glob(sd+"/*.lock")+glob.glob(sd+"/sessions.json"):
        try: os.remove(f)
        except OSError: pass
procs=[]
for i,s in enumerate(samples):
    for cond in ("w","wo"):
        agent=f"{model}_{cond}{i}"
        cell=f"{ROOT}/results/cells/{agent}"; ws=f"{ROOT}/ws/{agent}"
        hygiene(agent)
        log=open(f"{ROOT}/results/cells/{agent}.{RUNTAG}log","w")
        p=subprocess.Popen(["python3",f"{ROOT}/run_cell_local.py",agent,cell,f"{ROOT}/tasks",
                            RUNTAG,str(s),ws,MODELS[model],TIMEOUT],env=env,stdout=log,stderr=subprocess.STDOUT)
        procs.append((agent,s,p))
man[model]={"model":MODELS[model],"samples":samples,"runtag":RUNTAG,
            "agents":{"with_skill":[f"{model}_w{i}" for i in range(NS)],
                      "without_skill":[f"{model}_wo{i}" for i in range(NS)]}}
json.dump(man,open(MAN,"w"),indent=2)
print(f"launched {len(procs)} simultaneous runs for {model}, samples={samples}",flush=True)
for agent,s,p in procs: p.wait()
open(f"{ROOT}/results/prun_{model}_DONE","w").write("done "+time.strftime("%F %T"))
print("model done:",model)
