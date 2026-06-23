import os
import subprocess, os, time
ROOT=os.path.expanduser('~/P26exp')
models=[('gpt55','openrouter/openai/gpt-5.5'),('claude','openrouter/anthropic/claude-opus-4.8'),
        ('gemini','openrouter/google/gemini-3.1-pro-preview'),('qwen','openrouter/qwen/qwen3.7-max'),
        ('deepseek','openrouter/deepseek/deepseek-v4-pro')]
jobs=[(f'{s}_{c}',mid) for s,mid in models for c in ['w','wo']]
MAX=3; TO='800'
env=dict(os.environ); env['PATH']='/opt/homebrew/bin:'+env.get('PATH','')
def launch(ag,mid):
    cell=f'{ROOT}/results/cells/{ag}'
    lf=open(f'{cell}.runlog','w')
    return subprocess.Popen(['python3',f'{ROOT}/run_cell.py',ag,cell,f'{ROOT}/tasks','clean2','0',f'{ROOT}/ws/{ag}',mid,TO],env=env,stdout=lf,stderr=subprocess.STDOUT)
running=[]
for ag,mid in jobs:
    while len([p for p in running if p.poll() is None])>=MAX: time.sleep(3)
    running.append(launch(ag,mid))
for p in running: p.wait()
open(f'{ROOT}/results/clean_DONE','w').write('done '+time.strftime('%H:%M:%S')+'\n')
