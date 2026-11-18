from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'harness'))
from token_accounting import aggregate


def test_invalid_tokens_are_in_the_all_run_mean_and_cost():
    rows=[dict(model='m',condition='with_skill',valid=True,correct=True,tokens=100,secs=2,cost_usd=.1),
          dict(model='m',condition='with_skill',valid=False,correct=False,tokens=300,secs=6,cost_usd=.3)]
    summary,invalid=aggregate(rows);s=summary[0]
    assert s['mean_tokens_all']==200
    assert s['mean_tokens_valid']==100
    assert s['tokens_invalid_known']==300
    assert s['mean_seconds_all']==4
    assert s['usd_per_correct']==.4
    assert len(invalid)==1


def test_unknown_failed_usage_is_not_fabricated_as_zero():
    rows=[dict(model='m',condition='x',valid=True,tokens=100,secs=1),
          dict(model='m',condition='x',valid=False,secs=9)]
    summary,_=aggregate(rows);s=summary[0]
    assert s['unknown_token_runs']==1
    assert s['mean_tokens_all']is None
    assert s['mean_tokens_all_lower_bound']==50
    assert s['invalid_runs_unknown_tokens']==1
    assert s['usd_per_correct']is None
