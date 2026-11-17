"""Shard only unfinished original Stage A audits; preserve prompts and completed rows."""
import os,sys,json,time,types,hashlib,argparse,datetime,subprocess
from pathlib import Path
R=Path('/storage_fast/ttu/GenPlaylist');E=R/'experiments/cue-v2-stageA-20261112';S=R/'outputs/cue-v2-stageA-20261112';J=R/'outputs/cue-v2-judge-parallel-20261113'
sys.path.insert(0,str(E));from common import atomic,sha

def run_part(part):
 plan=json.loads((J/'plan.json').read_text());assigned={tuple(x) for x in plan['parts'][part]};source=(E/'diagnostics.py').read_text();needle=' if not jobs:return';assert source.count(needle)==1
 source=source.replace(needle," jobs=[j for j in jobs if tuple(j[:3]) in assigned]\n"+needle)
 m=types.ModuleType('original_judge_shard');m.__dict__.update(__name__='original_judge_shard',assigned=assigned);exec(compile(source,str(E/'diagnostics.py'),'exec'),m.__dict__)
 original=m.atomic
 def atomic_redirect(p,x):
  if Path(p)==S/'judge-complete.json':return
  assert Path(p).is_relative_to(S/'mpd') or Path(p).is_relative_to(S/'m4a'),p
  return original(p,x)
 m.atomic=atomic_redirect;t=time.time();m.judge()
 for ds,a,i in assigned:assert (S/ds/'judge'/a/(i+'.json')).exists()
 atomic(J/f'part-{part}-complete.json',dict(part=part,records=len(assigned),elapsed_seconds=time.time()-t,original_script_sha256=sha(E/'diagnostics.py'),plan_sha256=sha(J/'plan.json')))

def finish():
 deadline=datetime.datetime.fromisoformat('2026-11-13T11:11:30+08:00').timestamp();plan=json.loads((J/'plan.json').read_text())
 while not all((J/f'part-{i}-complete.json').exists() for i in range(3)):
  if time.time()>deadline:raise TimeoutError('Lease ended before audits completed; no false completion')
  time.sleep(20)
 import diagnostics
 count=0
 for ds in ['mpd','m4a']:
  co,arms=diagnostics.arms(ds)
  for a,items in arms.items():
   for i,cues in items.items():
    p=S/ds/'judge'/a/(i+'.json');x=json.loads(p.read_text());assert x['cues']==cues and x['dataset']==ds and x['arm']==a and x['item_id']==i;assert sorted(j['index'] for j in x['judgments'])==list(range(8));count+=1
 for p,h in plan['preserved_sha256'].items():assert sha(S/p)==h,'Existing audit overwritten'
 assert count==600
 parts=[json.loads((J/f'part-{i}-complete.json').read_text()) for i in range(3)];summary=dict(records=600,preserved=len(plan['preserved_sha256']),parallel_completed=sum(x['records'] for x in parts),parts=parts,completed_epoch=time.time(),protocol_unchanged=True)
 atomic(J/'combined-verification.json',summary);atomic(S/'judge-complete.json',summary);atomic(S/'done/judge.json',dict(elapsed_seconds=time.time()-plan['started_epoch'],started_epoch=plan['started_epoch'],script_sha256=sha(Path(__file__)),command=['parallel-original-judge','three-disjoint-parts'],evidence=str(J/'combined-verification.json')))
 import worker
 worker.run('human-audit','main','diagnostics.py',['export'])
 worker.run('stage-A-verification-four-arm','ace-step','verify.py')
 worker.run('stage-A-delivery','main','export.py')
 atomic(J/'finished.json',dict(completed_epoch=time.time(),stage_B_started=False,paper_changed=False,pushed=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--part',type=int);p.add_argument('--finish',action='store_true');a=p.parse_args()
 try:finish() if a.finish else run_part(a.part)
 except BaseException as e:
  import traceback
  atomic(J/('finish-failed.json' if a.finish else f'part-{a.part}-failed.json'),dict(traceback=traceback.format_exc(),epoch=time.time()));raise
