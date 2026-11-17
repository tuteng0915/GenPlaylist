"""CPU-only source audio verification and categorized existing-text screening."""
import collections
from common import *
def main():
 import soundfile as sf
 started=time.time();result={}
 for ds in ['mpd','m4a']:
  co=json.loads((O/ds/'cohort.json').read_text());mapping=json.loads((O/ds/'audio-paths.json').read_text());required={r['condition_item'] for r in co['gate_rows']};audit=[];failed=[]
  # MPD availability is an explicit deliverable; verify all located MPD audio.
  selected=sorted(mapping if ds=='mpd' else required)
  for i in selected:
   p=Path(mapping[i])
   try:
    info=sf.info(p)
    with sf.SoundFile(p) as f:w=f.read(min(info.samplerate,info.frames),dtype='float32')
    import numpy as np
    assert info.duration>=30 and w.size>0 and np.isfinite(w).all()
    audit.append(dict(item=i,duration=info.duration,frames=info.frames,sample_rate=info.samplerate,bytes=p.stat().st_size,sha256=sha(p) if i in required else None))
   except Exception as e:failed.append(dict(item=i,error=str(e)))
  result[ds]=dict(located=len(mapping),checked=len(selected),passed=len(audit),failed=failed,scope='all located MPD files; all required M4A first-target reference files',check='header/duration plus first-second finite decode; full decoding is performed when extracting metrics',records=audit)
  assert not (required & {f['item'] for f in failed}),f'{ds} reference decoding failed'
 atomic(O/'audio-source-audit.json',dict(elapsed_seconds=time.time()-started,datasets=result))
 scan=json.loads((O/'offensive-generated-text-scan.json').read_text());categories={'racial_slur':re.compile(r'\bnigg(?:a|as|er|ers)\b',re.I),'homophobic_slur':re.compile(r'\bfagg?(?:ot|ots)\b',re.I),'other_profanity':re.compile(r'\b(?:cunts?|motherfuck\w*|fuck\w*|bitch(?:es)?|sluts?|whores?)\b',re.I)};out={}
 for key,g in scan['groups'].items():
  assert g['records']==g['expected'],key
  hits={c:[] for c in categories}
  for record in g['hits']:
   p=Path(record['path']);assert sha(p)==record['sha256'];d=json.loads(p.read_text())
   for c,regex in categories.items():
    fields={f:len(regex.findall(str(d.get(f,'')))) for f in ['lyric_draft','music_attributes']}
    if any(fields.values()):hits[c].append(dict(id=record['id'],path=record['path'],fields=fields))
  out[key]=dict(records=g['records'],any_screen_hit_records=g['hit_records'],category_hit_records={k:len(v) for k,v in hits.items()},hits=hits)
 atomic(O/'offensive-categorized-scan.json',dict(records=scan['total_records'],lexical_screen_not_contextual_adjudication=True,record_counts_not_unique_lyrics=True,groups=out,elapsed_seconds=time.time()-started))

if __name__=='__main__':main()
