import sys, os, json, subprocess, glob, time
ROOT=os.environ.get("P26_ROOT",os.path.expanduser("~/P26exp")); HOME=os.path.expanduser("~")
MODELS={"deepseek":"deepseek/deepseek-v4-pro","kimi":"moonshot/kimi-k2.6"}
RUNTAG="prun"; TIMEOUT="600"
MAN=f"{ROOT}/results/parallel_manifest.json"
model=sys.argv[1]
man=json.load(open(MAN)); samples=man[model]["samples"]
env=dict(os.environ); env["PATH"]="/opt/homebrew/bin:"+env.get("PATH","")
subprocess.run(["openclaw","hooks","disable","session-memory"],env=env)
def hygiene(agent):
    sd=f"{HOME}/.openclaw/agents/{agent}/sessions"
    for f in glob.glob(sd+"/*.jsonl")+glob.glob(sd+"/*.lock")+glob.glob(sd+"/sessions.json"):
        try: os.remove(f)
        except OSError: pass
# paired 2-concurrency: each sample's w+wo together, sample-pairs sequential
for i,s in enumerate(samples):
    procs=[]
    for cond in ("w","wo"):
        agent=f"{model}_{cond}{i}"; cell=f"{ROOT}/results/cells/{agent}"; ws=f"{ROOT}/ws/{agent}"
        hygiene(agent)
        log=open(f"{ROOT}/results/cells/{agent}.{RUNTAG}log","w")
        p=subprocess.Popen(["python3",f"{ROOT}/run_cell_local.py",agent,cell,f"{ROOT}/tasks",
                            RUNTAG,str(s),ws,MODELS[model],TIMEOUT],env=env,stdout=log,stderr=subprocess.STDOUT)
        procs.append(p)
        print(f"launched pair sample {s}: {agent}",flush=True)
    for p in procs: p.wait()
    print(f"pair sample {s} done",flush=True)
open(f"{ROOT}/results/prun2_{model}_DONE","w").write("done "+time.strftime("%F %T"))
print("MODEL DONE",model)
