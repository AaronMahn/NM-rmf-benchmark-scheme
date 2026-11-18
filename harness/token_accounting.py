"""Aggregate all-run token/time/cost use, keeping invalid and unknown runs visible."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def normalized(row):
    usage = row.get('usage') or {}
    inp = row.get('input_tokens', usage.get('input_tokens', usage.get('prompt_tokens')))
    out = row.get('output_tokens', usage.get('output_tokens', usage.get('completion_tokens')))
    total = row.get('tokens')
    if not isinstance(total, (int,float)):
        total = inp+out if isinstance(inp,(int,float)) and isinstance(out,(int,float)) else None
    return dict(model=row.get('model') or row.get('intended_model'),
        condition=row.get('condition') or row.get('mode'),
        sample=row.get('sample', row.get('task')), agent=row.get('agent',row.get('case_id')),
        valid=bool(row.get('valid',row.get('valid_six_vector',False))),
        correct=row.get('correct'), tokens=total,input_tokens=inp,output_tokens=out,
        secs=row.get('secs',row.get('wall_seconds')),
        cost_usd=row.get('cost_usd',row.get('actual_cost_usd',usage.get('cost'))))


def aggregate(rows):
    rows=[normalized(row) for row in rows]
    groups=defaultdict(list)
    for row in rows:groups[(row['model'],row['condition'])].append(row)
    summary=[]
    for (model,condition), group in sorted(groups.items()):
        known=[r for r in group if r['tokens'] is not None]
        good=[r for r in group if r['valid']]
        invalid=[r for r in group if not r['valid']]
        costs=[r['cost_usd']for r in group if r['cost_usd'] is not None]
        times=[r['secs']for r in group if r['secs'] is not None]
        total=sum(r['tokens']for r in known)
        ncorrect=sum(r['correct'] is True for r in group)
        summary.append(dict(model=model,condition=condition,n=len(group),n_valid=len(good),
            n_invalid=len(invalid),known_token_runs=len(known),unknown_token_runs=len(group)-len(known),
            tokens_total_known=total,tokens_invalid_known=sum(r['tokens']for r in invalid if r['tokens']is not None),
            invalid_runs_unknown_tokens=sum(r['tokens']is None for r in invalid),
            mean_tokens_all=total/len(group) if len(known)==len(group) else None,
            mean_tokens_all_lower_bound=total/len(group),
            sd_tokens_all=statistics.stdev(r['tokens']for r in known) if len(known)==len(group) and len(group)>1 else None,
            mean_tokens_valid=statistics.mean(r['tokens']for r in good) if good and all(r['tokens']is not None for r in good) else None,
            mean_seconds_all=statistics.mean(times) if len(times)==len(group) else None,
            total_seconds_known=sum(times),cost_usd_known=sum(costs),unknown_cost_runs=len(group)-len(costs),
            n_correct=ncorrect,usd_per_correct=sum(costs)/ncorrect if ncorrect and len(costs)==len(group) else None))
    return summary,[r for r in rows if not r['valid']]


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        if not rows:return
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'results/par_all.json')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();data=json.loads(args.results.read_text())
    rows=data if isinstance(data,list) else data.get('rows',data.get('runs'))
    if not isinstance(rows,list):raise ValueError('Expected a rows/runs list')
    summary,invalid=aggregate(rows);args.out.mkdir(parents=True,exist_ok=True)
    write_csv(args.out/'all_run_accounting.csv',summary)
    write_csv(args.out/'invalid_run_tokens.csv',invalid)
    report={'input_rows':len(rows),'invalid_rows':len(invalid),'groups':summary,'invalid_runs':invalid}
    (args.out/'accounting.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'input_rows':len(rows),'invalid_rows':len(invalid),
        'tokens_total_known':sum(r['tokens_total_known']for r in summary),
        'tokens_invalid_known':sum(r['tokens_invalid_known']for r in summary)}))


if __name__=='__main__':main()
