"""User-directed removal of the unstarted history-cue SID-only branch."""
from pathlib import Path
import ast, copy, hashlib, json

W = Path(__file__).resolve().parent
E = W.parent / 'tiger-joint-20261108'
def put(name, text):
    if name.endswith('.py'): ast.parse(text)
    (W/name).write_text(text)

s = (E/'audio.sh').read_text()
s = s.replace('SID=$O/$DS/sid/audio/plan.json', 'SID=$R/outputs/two-dataset-fair-20261108/$DS/tiger/audio-plan.json')
s = s.replace('case $ARM in full) SYSTEM=GenPlaylist;; sft) SYSTEM=DDBC-SFT;; *) exit 2;; esac', '[[ $ARM == full ]] || exit 2\nSYSTEM=GenPlaylist')
put('audio_joint_v2.sh', s)
s = (E/'pair.sh').read_text().replace('for ARM in full sft; do bash "$E/audio.sh" "$DS" "$ARM" "$START" "$COUNT"; done', 'bash "$E/audio_joint_v2.sh" "$DS" full "$START" "$COUNT"').replace('validate_audio.py', 'validate_joint_v2.py')
put('joint_v2.sh', s)
s = (E/'validate_audio.py').read_text().replace("[('full','joint'),('sft','sid')]", "[('full','joint')]").replace("[('full','GenPlaylist'),('sft','DDBC-SFT')]", "[('full','GenPlaylist')]")
s = s.replace("'histories':a.count", "'histories':a.count,'scope':'joint-only; standard SID baseline is reused from verified two-dataset outputs'").replace('Validated paired TIGER audio', 'Validated joint TIGER audio')
put('validate_joint_v2.py', s)
s = (E/'report.py').read_text()
start = s.index("if a.mode=='matched':")
end = s.index("if a.mode=='catalog':", start)
s = s[:start] + s[end:]
s = s.replace("choices=['matched','catalog','audio']", "choices=['catalog','audio']")
s = s.replace("for arm in ['sid','joint']:", "for arm in ['joint']:")
needle = " if a.dataset=='mpd':plans['genplaylist']="
replacement = " baseline_path=B/a.dataset/'tiger/full-plan.json'\n baseline=json.loads(baseline_path.read_text());assert baseline['checkpoint_steps']==20000 and baseline['native_beam_width']==20 and baseline['allow_visible_history']\n plans['sid']=baseline['predictions']\n" + needle
assert needle in s
s = s.replace(needle, replacement, 1)
s = s.replace("cues=p['cue_ids'] if arm!='sid' else None", "cues=p['cue_ids'] if arm!='sid' else [[item_cues[str(song)][:8] for song in row] for row in p['item_ids']]")
s = s.replace("'sid':O/'audio/sft'", "'sid':B/a.dataset/'audio/tiger'")
s = s.replace("save('catalog-report.json'", "save('catalog-standard-baseline-report.json'").replace("save('audio-report.json'", "save('audio-standard-baseline-report.json'")
s = s.replace("'scope':'follow-up fixed-checkpoint comparisons, matched TIGER updates not diffusion FLOPs; same-context SID control is TIGER-HCue, not native TIGER'", "'baseline_plan_sha256':sha(baseline_path),'scope':'standard TIGER SID->SID versus joint SID+cue->SID+cue. Both fixed20k checkpoints, same test cases/catalog/beam20/unique-list policy; not identical initialization or training implementation. Standard TIGER retrieved-cue grounding is 1 by construction.'")
s = s.replace("'interpretation':'CLAP", "'baseline':'standard TIGER SID->SID; its catalog cues enter only the downstream renderer, never its recommender input','interpretation':'CLAP")
put('report_standard_v2.py', s)

before=json.loads((W/'tasks.before.json').read_text())
removed=[];after=[]
for old in before:
    t=copy.deepcopy(old);name=t['id']
    if name.startswith('tjc-'):
        if '-infer-sid-' in name or name.endswith('-metrics-sft') or name.endswith('-matched'):
            removed.append(t);continue
        ds=name.split('-')[1]
        t['cmd']=t['cmd'].replace('/pair.sh ', '/joint_v2.sh ').replace('/report.py ', '/report_standard_v2.py ')
        t['deps']=[d for d in t['deps'] if '-infer-sid-' not in d and not d.endswith('-metrics-sft')]
        t['deps']=[f'tjc-{ds}-train-joint' if d==f'tjc-{ds}-matched' else d for d in t['deps']]
    after.append(t)
assert len(before)==1615 and len(after)==1607 and len(removed)==8
ids={t['id'] for t in after};assert len(ids)==len(after)
assert all(d in ids for t in after for d in t.get('deps',[]))
graph={t['id']:t.get('deps',[]) for t in after};seen=set();active=set()
def visit(n):
    assert n not in active, n
    if n in seen:return
    active.add(n)
    for d in graph[n]:visit(d)
    active.remove(n);seen.add(n)
for n in graph:visit(n)
assert [t for t in before if not t['id'].startswith('tjc-')]==[t for t in after if not t['id'].startswith('tjc-')]
payload=(json.dumps(after,indent=2)+'\n').encode();(W/'tasks.after.json').write_bytes(payload)
manifest={'reason':'User explicitly requires SID-only input/output baseline; cancel auxiliary TIGER history-cue-input SID-output follow-ups before inference or audio starts.', 'before_sha256':hashlib.sha256((W/'tasks.before.json').read_bytes()).hexdigest(),'after_sha256':hashlib.sha256(payload).hexdigest(),'before_count':len(before),'after_count':len(after),'removed_tasks':removed,'changed_task_ids':[t['id'] for t in after if t!=next(x for x in before if x['id']==t['id'])],'preserved':'All completed training/checkpoints, all non-TIGER tasks and ordering; original protocols/scripts are archival. New joint-only audio keeps renderer/seeds/history/protocol.'}
(W/'change-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ['before_count','after_count','before_sha256','after_sha256']}))
