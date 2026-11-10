"""Atomic revision of only unstarted tasks, guarded against concurrent execution."""
import argparse, fcntl, hashlib, json, os
from pathlib import Path
from datetime import datetime
p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
R=Path('/storage_fast/ttu/GenPlaylist');W=R/'experiments/tiger-standard-correction-20261110'
Q=R/'experiments/two-dataset-fair-20261108/tasks.json';J=R/'outputs/two-dataset-fair-20261108/jobs'
m=json.loads((W/'change-manifest.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q)==m['before_sha256'];assert sha(W/'tasks.after.json')==m['after_sha256']
changed=m['changed_task_ids']+[t['id'] for t in m['removed_tasks']]
locks=[]
for name in changed:
    f=(J/(name+'.lock')).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    assert not any((J/(name+s)).exists() for s in ['.done.json','.failed.json','.running.json']),name
running=[]
for f in J.glob('*.running.json'):
    d=json.loads(f.read_text());assert not d['task'].startswith('tjc-'),d
    with (J/(d['task']+'.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:running.append(d)
        else:raise AssertionError('Stale running marker: '+d['task'])
assert len(running)==4 and {d['gpu'] for d in running}=={'0','1','2','3'}
baselines={}
for ds,n in [('mpd',941),('m4a',19771)]:
    root=R/'outputs/two-dataset-fair-20261108'/ds
    full=json.loads((root/'tiger/full-plan.json').read_text());audio=json.loads((root/'tiger/audio-plan.json').read_text())
    assert len(full['predictions']['item_ids'])==n and len(audio['predictions']['item_ids'])==941
    assert full['checkpoint_steps']==20000 and full['native_beam_width']==20 and full['allow_visible_history']
    for plan in [full,audio]:
        assert all(len(row)==5 and len(set(row))==5 for row in plan['predictions']['item_ids'])
    metrics=root/'audio/tiger/metrics'
    assert len(list((metrics/'mert-rows').glob('*.npz')))==941
    assert json.loads((metrics/'clap/ddbc_sft_clap.json').read_text())['examples']==941
    assert json.loads((metrics/'fad.json').read_text())['examples_per_corpus']==941
    baselines[ds]={'full_histories':n,'audio_histories':941,'full_plan_sha256':sha(root/'tiger/full-plan.json'),'audio_plan_sha256':sha(root/'tiger/audio-plan.json')}
result={'checked_sgt':datetime.now().astimezone().isoformat(),'status':'preflight-passed','baselines':baselines,'running_tasks_unchanged':[d['task'] for d in running],'queue_before_sha256':m['before_sha256'],'queue_after_sha256':m['after_sha256'],'queue_count':m['after_count']}
if a.apply:
    backup=W/'tasks.original-1615.json';assert not backup.exists();backup.write_bytes(Q.read_bytes())
    # Workers finish their current unrelated task before re-reading the queue.
    # Preserve task locks and current commands; no process restart or fake completion.
    assert sha(Q)==m['before_sha256']
    temp=Q.with_name('tasks.standard-sid.next.json');temp.write_bytes((W/'tasks.after.json').read_bytes());os.replace(temp,Q)
    assert sha(Q)==m['after_sha256'];result['status']='applied-and-verified'
    (W/'applied.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
