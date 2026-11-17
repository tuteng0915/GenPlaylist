"""Batched text speed/equivalence check; never mutates renderer inputs or renders audio."""
import json,sys,time,os
from pathlib import Path
import numpy as np
R=Path('/storage_fast/ttu/GenPlaylist');S=R/'outputs/cue-v2-stageA-20261112';O=R/'outputs/cue-v2-r31c-20261113/text-benchmark'
sys.path.insert(0,str(R/'experiments/cue-v2-stageA-20261112'))
from common import atomic,immutable,module,paths,sha,MODEL,REV
CODE=R/'v2-recovery-20261106/code'

def main():
    started=time.time();v=module('batch_frozen_verbal',CODE/'scripts/run_end_to_end_verbalization.py');vb=v._load_verbalization_module();jobs=[]
    for ds in ['mpd','m4a']:
        x=paths(ds);meta=v.CatalogItem.load_catalog(str(x['data']/'catalog_metadata.json'));meta={m.item_id:m for m in meta} if isinstance(meta,list) else meta
        lookup={str(i):int(j) for i,j in json.loads((x['art']/'item_id_to_row.json').read_text()).items()};emb=np.load(x['art']/'catalog_item_embeddings.npy').astype(np.float32)
        for arm in ['v1','none']:
            for i in [0,53,117,199]:
                p=S/ds/'gate'/arm/'verbalization/GenPlaylist'/f'{i:04d}.json';rec=json.loads(p.read_text());_,sigma=v._reference_structure(rec['reference_item_ids'],lookup,emb)
                nb=[meta[t] for t in rec['neighbor_item_ids']];style=[meta[t] for t in rec['style_item_ids']];kw=dict(reference_music=[meta[t] for t in rec['reference_item_ids']],cue_terms=rec['cue_terms'],user_instruction='')
                captured=[]
                def capture(prompt,system=''):
                    captured.append(dict(prompt=prompt,system=system));return 'CAPTURE_ONLY'
                vb.generate_music_attributes(nb,style,sigma,llm_call=capture,**kw)
                vb.generate_lyrics(nb,style,rec['music_attributes'],sigma,llm_call=capture,**kw)
                jobs.append(dict(dataset=ds,arm=arm,index=i,source_sha256=sha(p),old_record=rec,attribute_prompt=captured[0],lyric_prompt_fixed_old_attribute=captured[1],nb=nb,style=style,sigma=sigma,kwargs=kw))
    from vllm import LLM,SamplingParams
    llm=LLM(model=str(MODEL),dtype='bfloat16',max_model_len=12288,gpu_memory_utilization=.85,enforce_eager=True,max_num_seqs=16,seed=42);tok=llm.get_tokenizer()
    def batch(ps,max_tokens):
        prompts=[tok.apply_chat_template([dict(role='system',content=p['system']),dict(role='user',content=p['prompt'])],tokenize=False,add_generation_prompt=True) for p in ps]
        t=time.time();rr=llm.generate(prompts,SamplingParams(temperature=0,seed=42,max_tokens=max_tokens),use_tqdm=False)
        return [r.outputs[0].text.strip() for r in rr],time.time()-t
    attrs,ta=batch([j['attribute_prompt'] for j in jobs],128)
    ps=[]
    for j,attr in zip(jobs,attrs):
        cap=[]
        def capture(prompt,system=''):
            cap.append(dict(prompt=prompt,system=system));return 'CAPTURE_ONLY'
        vb.generate_lyrics(j['nb'],j['style'],attr,j['sigma'],llm_call=capture,**j['kwargs']);ps.append(cap[0])
    lyrics,tl=batch(ps,384)
    fixed,tf=batch([j['lyric_prompt_fixed_old_attribute'] for j in jobs],384)
    rows=[]
    for j,a,l,f,lp in zip(jobs,attrs,lyrics,fixed,ps):
        row={k:j[k] for k in ['dataset','arm','index','source_sha256','attribute_prompt','lyric_prompt_fixed_old_attribute']};old=j['old_record'];row.update(batched_attribute=a,batched_lyrics=l,lyric_prompt=lp,attribute_identical=a==old['music_attributes'],lyrics_identical=l==old['lyric_draft'],fixed_attribute_lyrics_identical=f==old['lyric_draft'],batched_fixed_attribute_lyrics=f);rows.append(row)
    immutable(O/'records.json',rows)
    exact=all(r['attribute_identical'] and r['lyrics_identical'] for r in rows)
    report=dict(examples=len(jobs),model_revision=REV,temperature=0,seed=42,dtype='bfloat16',max_tokens=[128,384],batch=16,attribute_seconds=ta,lyric_seconds=tl,steady_state_text_pairs_per_card_hour=len(jobs)*3600/(ta+tl),elapsed_seconds=time.time()-started,fixed_attribute_lyric_seconds=tf,attribute_exact=sum(r['attribute_identical'] for r in rows),lyric_exact=sum(r['lyrics_identical'] for r in rows),fixed_attribute_lyric_exact=sum(r['fixed_attribute_lyrics_identical'] for r in rows),sample_exact_equivalence=exact,can_claim_global_equivalence=False,adoption='pending; nonidentical backend outputs must not be silently substituted in comparisons with frozen controls',ace_clips_per_card_hour_previously_measured=407.536,estimated_serial_pairs_plus_ace_per_card_hour=1/(1/(len(jobs)*3600/(ta+tl))+1/407.536),audio_rendered=0)
    atomic(O/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        atomic(O/'failure.json',dict(traceback=traceback.format_exc()));raise
