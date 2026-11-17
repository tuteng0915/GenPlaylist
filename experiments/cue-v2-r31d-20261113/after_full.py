"""Finite, lease-bound follow-on draft/mapping diagnostics. Never train on a failed gate."""
import datetime,json,subprocess,time,fcntl,os
from pathlib import Path
R=Path('/storage_fast/ttu/GenPlaylist');O=R/'outputs/cue-v2-r31d-20261113';E=R/'experiments/cue-v2-r31d-20261113'
deadline='2026-11-13T11:11:30+08:00';end=datetime.datetime.fromisoformat(deadline).timestamp()
lock=open('/tmp/genplaylist-r31d-after-full.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

def save(name,d):
    p=O/name;t=p.with_suffix(f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)

try:
    complete=[O/'full'/d/f'complete-{k}.json' for d in ['mpd','m4a'] for k in range(4)]
    while not all(p.exists() for p in complete):
        failures=list((O/'full').glob('*/failure-*.json'))
        if failures:raise RuntimeError('Full extraction failure: '+str(failures))
        if time.time()>end-1800:raise TimeoutError('Reservation margin; preserve full extraction and wait for new booking')
        time.sleep(15)
    # Wait for the full worker's own engine and locks to be released. Never kill a holder here.
    while subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        if time.time()>end-1800:raise TimeoutError('No idle card inside current booking')
        time.sleep(5)
    time.sleep(5)
    tasks=[('build_vocab.py','main',O/'vocabulary-draft/complete.json'),('name_vocab.py','cue-v2-vllm',O/'vocabulary-draft/naming-complete.json'),('map_diagnose.py','main',O/'gate1.json'),('personalization.py','main',O/'personalization.json'),('benchmark_text.py','cue-v2-vllm',O/'text-benchmark/report.json')]
    for script,env,done in tasks:
        if done.exists():continue
        if time.time()>end-600:raise TimeoutError('New stage would exceed booking margin')
        name='r31d-'+Path(script).stem
        cmd=[str(R/'envs/main/bin/python'),str(R/'experiments/cue-v2-stageA-20261112/guard.py'),'--gpu','0','--name',name,'--deadline',deadline,'--',str(R/'envs'/env/'bin/python'),str(E/script)]
        save('follow-on-state.json',dict(stage=script,command=cmd,started=datetime.datetime.now().astimezone().isoformat(),booking='2204643781'))
        subprocess.run(cmd,stdout=open(O/(script+'.log'),'ab'),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True)
        assert done.exists(),str(done)
    gate=json.loads((O/'gate1.json').read_text())
    save('follow-on-complete.json',dict(gate1_passed=gate['passed'],training_started=False,next_step='Start author-authorized four-model training and400gateaudio only if gate1 passed; otherwise report failure without lowering criteria',completed=datetime.datetime.now().astimezone().isoformat()))
except BaseException:
    import traceback
    save('follow-on-failure.json',dict(traceback=traceback.format_exc(),preserve_all_outputs=True));raise
