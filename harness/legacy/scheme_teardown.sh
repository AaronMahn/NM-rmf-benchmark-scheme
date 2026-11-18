#!/usr/bin/env bash
# =============================================================================
# P26 benchmark SCHEME teardown  (Part III: start freshly)
# Run on the remote Mac. DESTRUCTIVE. Requires explicit confirmation:
#     bash scheme_teardown.sh --yes
#
# ONLY run this AFTER scheme_collect.py has produced results/scheme_bundle/ and
# you have pulled it to the local project folder.
#
# What it resets (the "settings made during the test process"):
#   1. stops all run_cell / openclaw-agent / benchmark screen processes
#   2. wipes every benchmark agent's session store (jsonl / lock / sessions.json)
#   3. clears results: results/cells/*, store.jsonl, scheme_* artifacts, DONE flags
#   4. clears per-cell workspace scratch (keeps skills/ and identity files)
#   5. removes the native 'deepseek' provider + deepseek API keys, and reverts the
#      deepseek_w/deepseek_wo model bindings back to OpenRouter
#      (backs up the configured openclaw.json first)
#   6. re-enables the session-memory hook (restore default)
#
# What it KEEPS: the harness scripts (run_cell.py etc.), the rmf-toolkit skill,
# tasks/, gt/ — those are the reusable scaffolding (also saved locally).
# =============================================================================
set -u
ROOT="${P26_ROOT:-$HOME/P26exp}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT="${P26_REPO:-$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)}"
OPENCLAW_HOME=$(PYTHONPATH="$REPO_ROOT/harness${PYTHONPATH:+:$PYTHONPATH}" \
 OPENCLAW_HOME="${OPENCLAW_HOME:-}" \
 python3 -c 'from runtime_config import OPENCLAW_HOME; print(OPENCLAW_HOME)')
export OPENCLAW_HOME
OPENCLAW_WORKSPACE_DIR=$(PYTHONPATH="$REPO_ROOT/harness${PYTHONPATH:+:$PYTHONPATH}" \
 OPENCLAW_HOME="$OPENCLAW_HOME" \
 python3 -c 'from runtime_config import OPENCLAW_WORKSPACE_DIR; print(OPENCLAW_WORKSPACE_DIR)')
OC=$OPENCLAW_HOME/openclaw.json

if [ "${1:-}" != "--yes" ]; then
  echo "This is destructive. Re-run with:  bash scheme_teardown.sh --yes"
  exit 1
fi

echo "[1/6] stopping processes + screens"
for s in $(screen -ls 2>/dev/null | grep -oE '[0-9]+\.(dw|dwo|dsw|dsre|dsmoke|ds|scheme)[a-z0-9]*'); do screen -S "$s" -X quit 2>/dev/null; done
pkill -f "run_cell.py"      2>/dev/null
pkill -f "scheme_run.py"    2>/dev/null
pkill -f "compute_nuclear|rmf_cal|compute_0" 2>/dev/null
sleep 2

echo "[2/6] wiping benchmark agent session stores"
for a in gpt55_w gpt55_wo claude_w claude_wo gemini_w gemini_wo qwen_w qwen_wo deepseek_w deepseek_wo kimi_w kimi_wo gpt54_w gpt54_wo; do
  rm -f "$OPENCLAW_HOME"/agents/$a/sessions/*.jsonl \
        "$OPENCLAW_HOME"/agents/$a/sessions/*.lock \
        "$OPENCLAW_HOME"/agents/$a/sessions/sessions.json 2>/dev/null
done

echo "[3/6] clearing results"
rm -rf $ROOT/results/cells/* $ROOT/results/scheme_bundle 2>/dev/null
rm -f  $ROOT/results/store.jsonl $ROOT/results/scheme_manifest.json \
       $ROOT/results/scheme_*_DONE $ROOT/results/clean_DONE 2>/dev/null

echo "[4/6] clearing per-cell workspace scratch (keep skills/ + identity files)"
KEEP="AGENTS.md BOOTSTRAP.md HEARTBEAT.md IDENTITY.md SOUL.md TOOLS.md USER.md MEMORY.md skills $OPENCLAW_WORKSPACE_DIR memory .git"
for ws in $ROOT/ws/*/; do
  [ -d "$ws" ] || continue
  for item in "$ws"*; do
    base=$(basename "$item")
    keep=0; for k in $KEEP; do [ "$base" = "$k" ] && keep=1; done
    [ $keep -eq 0 ] && rm -rf "$item"
  done
done

echo "[5/6] reverting openclaw.json (remove deepseek provider + revert bindings) + removing deepseek keys"
cp "$OC" "$OC.preteardown.bak.$(date +%s)"
python3 - "$OC" << 'PY'
import json,sys
p=sys.argv[1]; d=json.load(open(p))
d.get("models",{}).get("providers",{}).pop("deepseek",None)
ags=d.get("agents")
defs=ags if isinstance(ags,list) else next((v for v in ags.values() if isinstance(v,list)),[]) if isinstance(ags,dict) else []
for a in defs:
    if isinstance(a,dict) and a.get("id") in ("deepseek_w","deepseek_wo"):
        a["model"]="openrouter/deepseek/deepseek-v4-pro"
json.dump(d,open(p,"w"),indent=2)
print("  openclaw.json: deepseek provider removed; deepseek_w/wo reverted to openrouter")
PY
# remove deepseek keys from agent auth-profiles + inline models.json copy
for a in deepseek_w deepseek_wo; do
  AP=$OPENCLAW_HOME/agents/$a/agent/auth-profiles.json
  MJ=$OPENCLAW_HOME/agents/$a/agent/models.json
  [ -f "$AP" ] && python3 -c "import json;p='$AP';d=json.load(open(p));d.get('profiles',{}).pop('deepseek:default',None);json.dump(d,open(p,'w'),indent=2)"
  [ -f "$MJ" ] && python3 -c "import json;p='$MJ';d=json.load(open(p));d.get('providers',{}).pop('deepseek',None);json.dump(d,open(p,'w'),indent=2)"
done

echo "[6/6] re-enabling session-memory hook (default) + restarting gateway"
openclaw hooks enable session-memory 2>/dev/null
pkill -9 -f "openclaw-gateway.*18790" 2>/dev/null   # auto-respawns clean

echo "DONE. Fresh state. (openclaw.json backed up as $OC.preteardown.bak.*)"
echo "NOTE: harness scripts, rmf-toolkit skill, tasks/ and gt/ were intentionally kept."
