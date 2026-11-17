"""Finite Stage A 3544-song repair only. Three evidence passes; no Stage B or audio."""
import sys,json,time,collections,os
from pathlib import Path
E=Path('/storage_fast/ttu/GenPlaylist/experiments/cue-v2-repair-r3-20261113');sys.path.insert(0,str(E))
from pilot_extract import R,S,MODEL,REV,GRAMMAR,SYSTEM,postfilter,evidence_units,paths,atomic,immutable,sha,normalized
from normalize_format import expand
from complete_candidate_pilot import EXTRA
from pilot_semantic_filter import PROMPT as AUDIT_PROMPT
O=R/'outputs/cue-v2-repair-r5-stageA-20261113'
FIRST=R/'outputs/cue-v2-repair-r3-format-20261113';SECOND=R/'outputs/cue-v2-repair-r4-pilot-20261113';AUDITED=R/'outputs/cue-v2-repair-r4-audited-20261113'
def path(phase,ds,i):return O/phase/ds/(i+'.json')
def normalize_parsed(parsed,meta):
 for cat,rows in parsed.items():
  for c in rows:
   p=normalized(c['phrase'])
   if p!=normalized(meta.get('title','')):c['phrase']=expand(p,cat)
 return parsed

def main():
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import GuidedDecodingParams
 started=time.time();jobs=[];source={};meta={};pilot=json.loads((SECOND/'config.json').read_text())['selected']
 for ds in ['mpd','m4a']:
  co=json.loads((S/ds/'cohort.json').read_text());inputs=json.loads((S/ds/'inputs.json').read_text());meta[ds]=json.loads((paths(ds)['data']/'catalog_metadata.json').read_text());source[ds]={'cohort_sha256':sha(S/ds/'cohort.json'),'inputs_sha256':sha(S/ds/'inputs.json'),'songs':len(co['extraction_ids'])}
  for i in co['extraction_ids']:jobs.append((ds,i,inputs[i]))
 assert len(jobs)==3544
 # Dataset-interleaved, fixed order. Preserve previously completed pilot rows byte-for-byte in separate references.
 jobs.sort(key=lambda j:(int(j[1]),j[0]));cfg=dict(scope='Only original3544StageA songs; NOT16000catalog/NOTStageB',model=str(MODEL),revision=REV,seed=42,temperature=0,source=source,phases=['category-separated evidence-ID extraction','one bounded completion for<10unique candidates on lyric songs','conservative separate-prompt evidence audit'],system_prompt=SYSTEM,completion_prompt=EXTRA,semantic_audit_prompt=AUDIT_PROMPT,grammar=GRAMMAR,min_confidence=.75,real_cue_budget=2047,audio_started=False,scripts={p.name:sha(p) for p in E.glob('*.py')})
 immutable(O/'run_config.json',cfg)
 for ds,ids in pilot.items():
  for i in ids:
   for phase,parent in [('pass1',FIRST),('pass2',SECOND),('audited',AUDITED)]:
    p=path(phase,ds,i)
    if not p.exists():
     q=parent/ds/'candidates'/f'{i}.json';x=json.loads(q.read_text());x['reused_stageA_pilot_source']=str(q);x['reused_source_sha256']=sha(q);atomic(p,x)
 llm=LLM(model=str(MODEL),dtype='bfloat16',max_model_len=12288,gpu_memory_utilization=.85,enforce_eager=True,max_num_seqs=8,seed=42);tok=llm.get_tokenizer()
 normal=SamplingParams(temperature=0,max_tokens=4096,seed=42,guided_decoding=GuidedDecodingParams(grammar=GRAMMAR))
 for phase in ['pass1','pass2','audited']:
  phase_start=time.time();pending=[]
  for ds,i,inp in jobs:
   if path(phase,ds,i).exists():continue
   prior=None if phase=='pass1' else json.loads(path('pass1' if phase=='pass2' else 'pass2',ds,i).read_text())
   if phase=='pass2' and (not inp['lyrics'] or len({c['phrase'] for c in prior['accepted']})>=10):atomic(path(phase,ds,i),dict(prior,completion_not_needed=True));continue
   if phase=='audited' and not prior['accepted']:atomic(path(phase,ds,i),dict(prior,semantic_rejected=[],audit_not_needed='no accepted candidates'));continue
   pending.append((ds,i,inp,prior))
  for start in range(0,len(pending),8):
   batch=pending[start:start+8];prompts=[];params=[]
   for ds,i,inp,prior in batch:
    if phase=='audited':
     n=len(prior['accepted']);grammar='root ::= '+json.dumps('{"supported": [')+' boolean'+(' '+json.dumps(', ')+' boolean')*(n-1)+' '+json.dumps(']}')+'\nboolean ::= "true" | "false"\n';param=SamplingParams(temperature=0,max_tokens=512,seed=42,guided_decoding=GuidedDecodingParams(grammar=grammar));system=AUDIT_PROMPT;data=dict(source={k:inp[k] for k in ['genre','tags','mood','lyrics','lyric_quality']},candidates=[dict(index=j,phrase=c['phrase'],category=c['category'],cited_evidence=c['evidence'],source=c['source']) for j,c in enumerate(prior['accepted'])])
    else:
     system=SYSTEM if phase=='pass1' else SYSTEM+'\n'+EXTRA;data=dict(title=inp['title'],lyric_quality=inp['lyric_quality'],evidence=evidence_units(inp));param=normal
     if prior is not None:data['already_accepted']=[c['phrase'] for c in prior['accepted']]
    prompts.append(tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':json.dumps(data,ensure_ascii=False)}],tokenize=False,add_generation_prompt=True));params.append(param)
   responses=llm.generate(prompts,params,use_tqdm=False)
   for (ds,i,inp,prior),r in zip(batch,responses):
    raw=r.outputs[0];atomic(O/'raw'/phase/ds/(i+'.json'),dict(raw=raw.text,finish_reason=raw.finish_reason,tokens=len(raw.token_ids)));assert raw.finish_reason!='length',(phase,ds,i,'stop; preserve truncated record')
    if phase=='audited':
     flags=json.loads(raw.text)['supported'];assert len(flags)==len(prior['accepted']) and all(isinstance(v,bool) for v in flags);x=dict(prior,accepted=[c for c,v in zip(prior['accepted'],flags) if v],semantic_rejected=[c for c,v in zip(prior['accepted'],flags) if not v],audit_raw=raw.text)
    else:
     parsed=normalize_parsed(json.loads(raw.text),meta[ds][i]);a,rejected=postfilter(parsed,inp,meta[ds][i],evidence_units(inp));x=dict(input=inp,evidence_units=evidence_units(inp),accepted=a,rejected=rejected,run_config_sha256=sha(O/'run_config.json'))
     if prior is not None:
      merged={c['phrase']:c for c in prior['accepted']}
      for c in a:
       if c['phrase'] not in merged:merged[c['phrase']]=c
      x=dict(prior,accepted=list(merged.values()),second_pass_rejected=rejected)
    atomic(path(phase,ds,i),x)
   atomic(O/'progress.json',dict(phase=phase,completed_this_phase=min(start+8,len(pending)),phase_pending_at_start=len(pending),phase_elapsed_seconds=time.time()-phase_start,total_elapsed_seconds=time.time()-started,stage_B_started=False));print('STAGE_A_CANDIDATES',phase,min(start+8,len(pending)),len(pending),time.time()-started,flush=True)
  atomic(O/(phase+'-complete.json'),dict(records=3544,elapsed_seconds=time.time()-phase_start))
 report={}
 for ds in ['mpd','m4a']:
  counts=collections.Counter()
  for d,i,inp in jobs:
   if ds!=d:continue
   q=json.loads(path('audited',ds,i).read_text());counts['songs']+=1;counts['accepted']+=len(q['accepted']);counts['semantic_rejected']+=len(q.get('semantic_rejected',[]))
   if inp['lyrics']:counts['lyric_songs']+=1;counts['candidate_capacity_slots']+=min(8,len({c['phrase'] for c in q['accepted']}))
  report[ds]=dict(counts=counts,candidate_capacity=counts['candidate_capacity_slots']/(8*counts['lyric_songs']))
 atomic(O/'candidates-complete.json',dict(datasets=report,total_songs=3544,elapsed_seconds=time.time()-started,final_vocabulary_not_yet_built=True,final90percent_gate_not_yet_tested=True,audio_started=False,stage_B_started=False));print('STAGE_A_CANDIDATES_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  atomic(O/'failure.json',dict(traceback=traceback.format_exc(),epoch=time.time()));raise
