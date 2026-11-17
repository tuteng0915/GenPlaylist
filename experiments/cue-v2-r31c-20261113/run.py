"""Versioned r3.1 extraction/rewrite. Pilot reuses raw r3 outputs, never resamples them."""
import argparse, collections, json, re, sys, time
from pathlib import Path
R=Path('/storage_fast/ttu/GenPlaylist'); S=R/'outputs/cue-v2-stageA-20261112'; OLD=R/'outputs/cue-v2-r3-author-pilot-20261113'; O=R/'outputs/cue-v2-r31c-20261113'
sys.path.insert(0,str(R/'experiments/cue-v2-r3-author-pilot-20261113'))
import pilot as old
from policy import copying_reason,tokens,test_policy
atomic,immutable,sha=old.atomic,old.immutable,old.sha
EXTRACT=old.PROMPT.replace('2-6 English words','1-3 English words').replace('a reusable paraphrase','a reusable abstract concept (for example heartbreak, nostalgia, emotional dependence), not a description of lyric wording or song structure; use a faithful paraphrase')
REWRITE='''Convert each supplied lyrical candidate to a concise reusable abstract theme of at most THREE English words. Use its cited evidence only. Music4All lyrics are stemmed/tokenized; infer conservatively, never restore an invented story. Paraphrase the concept, not a lyric fragment: "drive all night long" -> "late-night driving"; "I am stronger than yesterday" -> "personal growth". Avoid names, titles, offensive/explicit words, filler, and commentary about repetition or song structure. One or two ordinary words may overlap lyrics; never copy three consecutive content words or an entire lyric line. Preserve distinct supported facets. Return exactly one entry for each candidate, in the same order, preserving its evidence_id. Empty phrase and confidence 0.0 if the evidence cannot support a meaningful concept; do not invent to meet a count. Output JSON {"themes":[{"phrase":"...","evidence_id":"L0","confidence":0.9}]}.'''
REWRITE_GRAMMAR=r'''
root ::= "{\"phrase\": " phrase ", \"confidence\": " confidence "}"
phrase ::= "\"" ([a-zA-Z]+ (" " [a-zA-Z]+){0,2})? "\""
confidence ::= "0.0" | "0.25" | "0.5" | "0.6" | "0.7" | "0.75" | "0.8" | "0.85" | "0.9" | "0.95" | "1.0"
'''+old.CHAR
REWRITE=REWRITE[:REWRITE.index('Return exactly one entry')]+'Return one JSON object only: {"phrase":"...","confidence":0.9}. The evidence is fixed externally. Empty phrase and confidence 0.0 if unsupported. Never fabricate a theme.'
TOPICS={'summer':'summer','christmas':'christmas','workout':'exercise','halloween':'halloween','winter':'winter','travel':'travel','road trip':'road travel','friendship':'friendship','heartbreak':'heartbreak','nostalgia':'nostalgia','love':'romantic love','breakup':'relationship breakup','protest':'social protest','war':'war','peace':'peace','nature':'nature','spiritual':'spirituality'}

def metadata(inp,meta,names):
    good,bad=old.metadata_candidates(inp,meta,names)
    # Describes usage/game tags rather than sound; the 40-song inspection found these.
    badsound={'guitar hero','makes me wanna dance'}
    retained=[]
    for c in good:
        if c['phrase'] in badsound:bad.append(dict(c,reason='nonmusical_usage_tag'))
        else:retained.append(c)
    if not inp['lyrics']:
        for tag in inp['tags']:
            p=TOPICS.get(old.norm(tag))
            if p and not old.forbidden(p,meta,names):
                if not any(c['category']=='theme' and c['phrase']==p for c in retained):
                    retained.append(dict(phrase=p,category='theme',source='tags',evidence=tag,confidence=1.0,construction='exact thematic tag normalization; no lyric inference'))
    return retained,bad

def inspect(c,inp,meta,names,ds):
    p=old.normalized(c['phrase']);u=old.evidence_units(inp).get(c['evidence_id'])
    why=old.forbidden(p,meta,names)
    if not 1<=len(tokens(p))<=3:why=why or 'length'
    if not u or u['source']!='lyrics':why=why or 'evidence_id'
    if u and old.normalized(u['text']) not in old.normalized(inp['lyrics']):why=why or 'evidence_quote'
    why=why or copying_reason(p,inp['lyrics'],ds)
    if c['confidence']<.75:why=why or 'low_confidence'
    if p in ['song lyrics','lyrical themes','human experience','life experiences','emotional feeling','repetition','repeated words','lyric repetition','repetitive lyrics']:why=why or 'generic_filler'
    return dict(phrase=p,category='theme',source='lyrics',evidence=u['text'] if u else '',evidence_id=c['evidence_id'],confidence=c['confidence']),why

def main(args):
    test_policy();started=time.time();co=json.loads((S/args.dataset/'cohort.json').read_text());inputs=json.loads((S/args.dataset/'inputs.json').read_text());meta=json.loads((old.paths(args.dataset)['data']/'catalog_metadata.json').read_text());names={old.norm(m.get('artist','')) for m in meta.values() if m.get('artist')}
    allids=co['pilot'] if args.scope=='pilot' else sorted(inputs)
    if args.scope=='full':
        g=json.loads((O/'gate0.json').read_text());assert g['passed'], 'Gate0 must pass before full extraction'
    ids=allids[args.shard::args.shards];base=O/args.scope/args.dataset
    cfg=dict(version='r3.1',dataset=args.dataset,scope=args.scope,ids=allids,model=str(old.MODEL),revision=old.REV,extract_prompt=EXTRACT,rewrite_prompt=REWRITE,grammar=old.GRAMMAR,rewrite_grammar=REWRITE_GRAMMAR,temperature=0,seed=42,max_tokens=1800,confidence_threshold=.75,policy_sha256=sha(Path(__file__).with_name('policy.py')),script_sha256=sha(Path(__file__)),source_script_sha256=sha(Path(old.__file__)),inputs_sha256=sha(S/args.dataset/'inputs.json'),cohort_sha256=sha(S/args.dataset/'cohort.json'),no_lyric_minimum=None,soft_quota=[2,2,4],pilot_raw_reuse='r3 raw candidate output; rewrite all over3word or copy candidates; no resampling',full_raw_policy='same evidence inputs; compact-theme prompt; same corrective rewrite policy')
    immutable(base/'config.json',cfg);pending=[]
    for i in ids:
        p=base/'candidates'/f'{i}.json'
        if p.exists():assert json.loads(p.read_text())['config_sha256']==sha(base/'config.json');continue
        if args.scope=='full' and (O/'pilot'/args.dataset/'candidates'/f'{i}.json').exists():
            row=json.loads((O/'pilot'/args.dataset/'candidates'/f'{i}.json').read_text());row['pilot_source_sha256']=sha(O/'pilot'/args.dataset/'candidates'/f'{i}.json');row['config_sha256']=sha(base/'config.json');atomic(p,row);continue
        a,b=metadata(inputs[i],meta[i],names)
        if not inputs[i]['lyrics']:atomic(p,dict(input=inputs[i],accepted=a,rejected=b,rewrites=[],config_sha256=sha(base/'config.json'),no_usable_lyrics=True));continue
        pending.append((i,a,b))
    if pending:
        from vllm import LLM,SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        llm=LLM(model=str(old.MODEL),dtype='bfloat16',max_model_len=12288,gpu_memory_utilization=.85,enforce_eager=True,max_num_seqs=16,seed=42);tok=llm.get_tokenizer()
        def generate(system,data,grammar,outpaths):
            prompts=[tok.apply_chat_template([dict(role='system',content=system),dict(role='user',content=json.dumps(x,ensure_ascii=False))],tokenize=False,add_generation_prompt=True) for x in data]
            params=SamplingParams(temperature=0,seed=42,max_tokens=(180 if grammar==REWRITE_GRAMMAR else 1800),guided_decoding=GuidedDecodingParams(grammar=grammar))
            rr=llm.generate(prompts,params,use_tqdm=False);result=[]
            for r,p in zip(rr,outpaths):
                z=r.outputs[0];atomic(p,dict(raw=z.text,finish_reason=z.finish_reason,output_tokens=len(z.token_ids)));assert z.finish_reason!='length',(str(p),'truncation; no silent resampling');result.append(json.loads(z.text) if grammar==REWRITE_GRAMMAR else json.loads(z.text)['themes'])
            return result
        t=time.time()
        for start in range(0,len(pending),16):
            batch=pending[start:start+16];raw=[];need=[]
            for idx,(i,a,b) in enumerate(batch):
                p=OLD/args.dataset/'raw'/f'{i}.json' if args.scope=='pilot' else base/'raw'/f'{i}.json'
                if p.exists():raw.append(json.loads(json.loads(p.read_text())['raw'])['themes'])
                else:raw.append(None);need.append(idx)
            if need:
                data=[dict(title_to_exclude=inputs[batch[k][0]]['title'],dataset=args.dataset,lyric_quality=inputs[batch[k][0]]['lyric_quality'],evidence={k:v['text'] for k,v in old.evidence_units(inputs[batch[k][0]]).items() if k.startswith('L')}) for k in need]
                outs=generate(EXTRACT,data,old.GRAMMAR,[base/'raw'/f'{batch[k][0]}.json' for k in need])
                for k,v in zip(need,outs):raw[k]=v
            rewrite_jobs=[];prepared=[]
            for (i,a,b),cands in zip(batch,raw):
                inp=inputs[i];keep=[];reject=[];rw=[]
                for idx,c in enumerate(cands):
                    row,why=inspect(c,inp,meta[i],names,args.dataset)
                    # Excluded names/offensive/title/evidence remain excluded. Length/copy gets one fixed rewrite.
                    if why in ['length','verbatim_content_trigram','verbatim_whole_line']:
                        rw.append((idx,c,why))
                    elif why:reject.append(dict(row,reason=why))
                    else:keep.append(row)
                prepared.append((i,a,b,keep,reject,rw))
                for j,(idx,c,why) in enumerate(rw):
                    rewrite_jobs.append((len(prepared)-1,j,dict(title_to_exclude=inp['title'],dataset=args.dataset,phrase=c['phrase'],evidence=old.evidence_units(inp).get(c['evidence_id'],{}).get('text',''),reason=why)))
            outs=generate(REWRITE,[d for _,_,d in rewrite_jobs],REWRITE_GRAMMAR,[base/'rewritten-raw'/f'{prepared[k][0]}-{j}.json' for k,j,_ in rewrite_jobs]) if rewrite_jobs else []
            rewrites={(k,j):dict(v,evidence_id=prepared[k][5][j][1]['evidence_id']) for (k,j,_),v in zip(rewrite_jobs,outs)}
            for k,(i,a,b,keep,reject,rw) in enumerate(prepared):
                trail=[]
                if rw:
                    for j,(idx,orig,why) in enumerate(rw):
                        new=rewrites[k,j]
                        assert new['evidence_id']==orig['evidence_id'],(i,'rewrite evidence changed')
                        new['confidence']=min(orig['confidence'],new['confidence'])
                        row,reason=inspect(new,inputs[i],meta[i],names,args.dataset)
                        trail.append(dict(index=idx,before=orig['phrase'],after=row['phrase'],trigger=why,remaining_exclusion=reason,evidence=row['evidence'],confidence=row['confidence']))
                        if reason:reject.append(dict(row,reason=reason,rewritten=True,original_phrase=orig['phrase']))
                        else:keep.append(dict(row,rewritten=True,original_phrase=orig['phrase']))
                seen=set();unique=[]
                for c in keep:
                    if c['phrase'] in seen:reject.append(dict(c,reason='duplicate'))
                    else:seen.add(c['phrase']);unique.append(c)
                atomic(base/'candidates'/f'{i}.json',dict(input=inputs[i],accepted=a+unique,rejected=b+reject,raw_theme_count=len(raw[k]),rewrites=trail,config_sha256=sha(base/'config.json')))
            atomic(base/f'progress-{args.shard}.json',dict(songs_done_in_this_run=min(start+16,len(pending)),pending_at_start=len(pending),elapsed_seconds=time.time()-started,inference_seconds=time.time()-t));print('R31',args.scope,args.dataset,args.shard,min(start+16,len(pending)),len(pending),time.time()-started,flush=True)
    atomic(base/f'complete-{args.shard}.json',dict(songs=len(ids),lyric_inferences_this_run=len(pending),elapsed_seconds=time.time()-started,scope=args.scope,shard=args.shard,shards=args.shards))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mpd','m4a'],required=True);p.add_argument('--scope',choices=['pilot','full'],required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    try:main(a)
    except BaseException:
        import traceback
        atomic(O/a.scope/a.dataset/f'failure-{a.shard}.json',dict(traceback=traceback.format_exc()));raise
