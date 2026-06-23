import json, os, shutil, subprocess, time
HOME=os.path.expanduser("~")
OC=f"{HOME}/.openclaw/openclaw.json"
ROOT=os.environ.get("P26_ROOT",os.path.expanduser("~/P26exp"))
MODELS=["gpt55","claude","gemini","qwen","deepseek","kimi"]
KEEP={"AGENTS.md","BOOTSTRAP.md","HEARTBEAT.md","IDENTITY.md","SOUL.md","TOOLS.md","USER.md","MEMORY.md","skills",".openclaw","memory",".git"}

shutil.copy(OC, OC+".preclone.bak")
d=json.load(open(OC))
ags=d.get("agents"); defs=ags if isinstance(ags,list) else next((v for v in ags.values() if isinstance(v,list)),[])
byid={a["id"]:a for a in defs if isinstance(a,dict)}
created=[]
for m in MODELS:
    for cond in ("w","wo"):
        base=f"{m}_{cond}"; b=byid[base]
        for i in (0,1,2):
            nid=f"{m}_{cond}{i}"
            nws=f"{ROOT}/ws/{nid}"; nad=f"{HOME}/.openclaw/agents/{nid}/agent"
            # agentDir (config + keys)
            os.makedirs(f"{HOME}/.openclaw/agents/{nid}/sessions", exist_ok=True)
            if os.path.isdir(nad): shutil.rmtree(nad)
            shutil.copytree(b["agentDir"], nad)
            # workspace copy (incl skills for _w)
            if os.path.isdir(nws): shutil.rmtree(nws)
            shutil.copytree(b["workspace"], nws, ignore=shutil.ignore_patterns("*.lock"))
            # clean scratch in workspace (keep identity + skills)
            for item in os.listdir(nws):
                if item not in KEEP:
                    p=os.path.join(nws,item); shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            entry={"id":nid,"name":nid,"workspace":nws,"agentDir":nad,"model":b["model"]}
            if nid in byid: byid[nid].update(entry)
            else: defs.append(entry); byid[nid]=entry
            created.append(nid)
json.dump(d, open(OC,"w"), indent=1)
print("created/updated agents:", len(created))
print("sample ids:", created[:6], "...", created[-3:])
# restart gateway to load
subprocess.run("pkill -9 -f 'openclaw-gateway' 2>/dev/null", shell=True); time.sleep(4)
print("gateway restarted")
