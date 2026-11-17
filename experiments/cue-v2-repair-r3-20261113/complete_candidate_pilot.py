"""One bounded second pass on insufficient pilot candidate pools; no catalog rebuild/audio."""
import sys,json,time,collections
from pathlib import Path
sys.path.insert(0,'/storage_fast/ttu/GenPlaylist/experiments/cue-v2-repair-r3-20261113')
from pilot_extract import R,S,O as FIRST,MODEL,REV,GRAMMAR,SYSTEM,postfilter,evidence_units,paths,atomic,immutable,sha
from normalize_format import D as BASE,expand
O=R/'outputs/cue-v2-repair-r4-pilot-20261113'
EXTRA='''This is a separate candidate-completion pass. Existing accepted descriptors are supplied; do not repeat them or offer near-synonyms. Find additional DISTINCT evidence-backed descriptors, especially thematic imagery, actions, places, relationships and feelings. Tokenized/stemmed lyric words are legitimate lexical evidence: describe what imagery/actions are explicitly present, without reconstructing an invented story. Do not treat grammatical damage alone as absence of all meaning. You may combine several supplied units in your reasoning, but cite the single best supporting evidence ID for each cue. A genre/tag supports its own musical style, not guessed instruments. Seek up to 8 additional candidates across categories when evidence supports them. This is not a requirement to fill all slots: never add unsupported detail. Use 2-5-word phrases even for emotions (for example 'expressed longing' rather than 'longing').'''

def main():
 import numpy as np
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import GuidedDecodingParams
 t=time.time();cfg=json.loads((FIRST/'pilot-config.json').read_text());jobs=[];existing={}
 for ds,ids in cfg['selected'].items():
  meta=json.loads((paths(ds)['data']/'catalog_metadata.json').read_text())
  for i in ids:
   x=json.loads((BASE/ds/'candidates'/f'{i}.json').read_text());existing[(ds,i)]=x
   if x['input']['lyrics'] and len({c['phrase'] for c in x['accepted']})<10:jobs.append((ds,i,x,meta[i]))
 immutable(O/'config.json',dict(version='r4-targeted-pilot-completion',selected=cfg['selected'],targeted=[[d,i] for d,i,_,_ in jobs],model=str(MODEL),revision=REV,system=SYSTEM+'\n'+EXTRA,grammar=GRAMMAR,temperature=0,seed=42,max_tokens=4096,minimum_confidence=.75,max_passes=1,scope='same40pilot songs, no3544rebuild/noaudio/noStageB'))
 llm=LLM(model=str(MODEL),dtype='bfloat16',max_model_len=12288,gpu_memory_utilization=.85,enforce_eager=True,max_num_seqs=8,seed=42);tok=llm.get_tokenizer();params=SamplingParams(temperature=0,max_tokens=4096,seed=42,guided_decoding=GuidedDecodingParams(grammar=GRAMMAR))
 for start in range(0,len(jobs),8):
  batch=jobs[start:start+8];prompts=[tok.apply_chat_template([{'role':'system','content':SYSTEM+'\n'+EXTRA},{'role':'user','content':json.dumps(dict(title=x['input']['title'],lyric_quality=x['input']['lyric_quality'],evidence=x['evidence_units'],already_accepted=[c['phrase'] for c in x['accepted']]),ensure_ascii=False)}],tokenize=False,add_generation_prompt=True) for ds,i,x,meta in batch];responses=llm.generate(prompts,params,use_tqdm=False)
  for (ds,i,x,meta),r in zip(batch,responses):
   v=r.outputs[0];atomic(O/ds/'raw-responses'/f'{i}.json',dict(raw=v.text,finish_reason=v.finish_reason,tokens=len(v.token_ids)));assert v.finish_reason!='length',(ds,i)
   parsed=json.loads(v.text)
   for cat,rows in parsed.items():
    for c in rows:
     if c['phrase'].lower()!=meta.get('title','').lower():c['phrase']=expand(c['phrase'].lower(),cat)
   a,rejected=postfilter(parsed,x['input'],meta,x['evidence_units']);merged={c['phrase']:dict(c,pass_index=1) for c in x['accepted']}
   for c in a:
    if c['phrase'] not in merged:merged[c['phrase']]=dict(c,pass_index=2)
   atomic(O/ds/'candidates'/f'{i}.json',dict(x,accepted=list(merged.values()),second_pass_rejected=rejected,parent_candidate_sha256=sha(BASE/ds/'candidates'/f'{i}.json')))
  print('COMPLETION_PASS',min(start+8,len(jobs)),len(jobs),time.time()-t,flush=True)
 for (ds,i),x in existing.items():
  p=O/ds/'candidates'/f'{i}.json'
  if not p.exists():atomic(p,dict(x,parent_candidate_sha256=sha(BASE/ds/'candidates'/f'{i}.json'),second_pass_not_needed=True))
 report={}
 for ds,ids in cfg['selected'].items():
  rows=[json.loads((O/ds/'candidates'/f'{i}.json').read_text()) for i in ids];valid=[x for x in rows if x['input']['lyrics']];cap=sum(min(8,len({c['phrase'] for c in x['accepted']})) for x in valid);report[ds]=dict(songs=len(rows),usable_lyrics_songs=len(valid),candidate_capacity_slots=cap,candidate_capacity=cap/(8*len(valid)),per_song=[dict(item=i,accepted=len(x['accepted']),usable=bool(x['input']['lyrics'])) for i,x in zip(ids,rows)])
 atomic(O/'candidate-capacity-report.json',dict(datasets=report,elapsed_seconds=time.time()-t,not_final_shared_vocabulary_coverage=True,semantic_faithfulness_pending=True,audio_started=False,stage_B_started=False));print('R4_PILOT_COMPLETE',flush=True)
if __name__=='__main__':main()
