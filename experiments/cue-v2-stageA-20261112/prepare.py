import argparse,collections,random
import numpy as np
from common import *

def stratified(ids,cat,quality,pop,n,seed):
 rng=random.Random(seed);ordered=sorted(ids);ranks={s:k/len(ids) for k,s in enumerate(sorted(ids,key=lambda s:(pop[s],s)))}
 groups=collections.defaultdict(list)
 for s in ordered:
  genre=normalized(cat[s].get('genre','')).split(',')[0] or 'unknown'
  groups[(genre,min(2,int(ranks[s]*3)),quality[s])].append(s)
 for g in groups.values():rng.shuffle(g)
 keys=sorted(groups);rng.shuffle(keys);result=[]
 while len(result)<n:
  for k in keys:
   if groups[k] and len(result)<n:result.append(groups[k].pop())
 return sorted(result)

def main():
 started=time.time();O.mkdir(parents=True,exist_ok=True);report={}
 for ds in ['mpd','m4a']:
  x=paths(ds);cp=R/'outputs/sampled-candidates-20261110'/ds/'candidates.json';co=json.loads(cp.read_text());cat=json.loads((x['data']/'catalog_metadata.json').read_text());ids=co['catalog_ids']
  cat={str(k):v for k,v in cat.items()};pop=collections.Counter()
  train=x['data']/'splits/train.txt'
  for line in train.read_text().splitlines():pop.update(line.split())
  inputs={i:evidence_input(ds,i,cat[i]) for i in ids};quality={i:d['lyric_quality'] for i,d in inputs.items()}
  pilot=stratified(ids,cat,quality,pop,100,20261112+(ds=='m4a'))
  pool=stratified(ids,cat,quality,pop,min(1000,len(ids)),20261113+(ds=='m4a'))
  pool=sorted(set(pool)|set(pilot))
  idx=sorted(np.random.default_rng(20261112+(ds=='m4a')).choice(941,200,replace=False).tolist())
  rows=[dict(local_index=k,source_history_index=i,history=co['histories'][i],targets=co['truth'][i],condition_item=co['truth'][i][0],seed=42000+i) for k,i in enumerate(idx)]
  gate_ids=sorted({s for row in rows for s in row['targets']})
  scope=sorted(set(pool)|set(gate_ids))
  immutable(O/ds/'cohort.json',dict(dataset=ds,pilot=pilot,vocab_pool=pool,gate_rows=rows,extraction_ids=scope,source_sha256=sha(cp),sampling_seed=20261112,gate='conditional target-item diagnostic; not end-to-end recommendation'))
  immutable(O/ds/'inputs.json',inputs)
  audio_roots=([Path('/storage/ttu/GenPlaylist/runtime/mpd/data/audio/spotify'),R/'original-ncl/data/audio/spotify',Path('/storage/ttu/GenPlaylist/data/audio/spotify')] if ds=='mpd' else [R/'original-ncl/data/audio/music4all-onion',R/'original-ncl/data/catalog-music4all-onion-v3-u500/audio'])
  audio_roots += [R/'outputs/list5-audio-20261111'/ds/'reference-audio']
  if ds=='m4a':audio_roots += [R/'outputs/two-dataset-fair-20261108/m4a/reference-audio']
  available={}
  for i in ids:
   candidates=[Path(cat[i].get('audio_path') or '/nonexistent')]+[root/(i+ext) for root in audio_roots for ext in ['.mp3','.wav','.flac']]
   found=next((p for p in candidates if p.is_file() and p.stat().st_size>0),None)
   if found:available[i]=str(found)
  atomic(O/ds/'audio-paths.json',available)
  instruments=re.compile(r'guitar|piano|violin|flute|drum|bass|sax|trumpet|cello|synth|banjo|organ',re.I)
  report[ds]=dict(catalog_size=len(ids),lyrics_quality=dict(collections.Counter(quality.values())),lyric_files=sum(v['lyrics_file'] is not None for v in inputs.values()),audio_files_located=len(available),audio_decode_not_yet_checked=True,metadata_instrument_mentions=sum(bool(instruments.search(' '.join(cat[i].get('tags',[]))+' '+cat[i].get('genre',''))) for i in ids),instrument_mentions_not_independently_verified=True,pilot_songs=100,vocab_pool=len(pool),gate_contexts=200,scope_songs=len(scope),gate_missing_audio=sorted({s for row in rows for s in row['targets']} - set(available)),source_hashes={str(p):sha(p) for p in [cp,x['data']/'catalog_metadata.json',x['cues']/'cue_vocab.json',x['cues']/'item2cues.json',train]})
  report[ds]['gate_missing_first_target_audio']=sorted({row['condition_item'] for row in rows}-set(available))
 atomic(O/'preflight.json',dict(started_epoch=started,elapsed_seconds=time.time()-started,datasets=report));print(json.dumps(report,indent=2),flush=True)

def scan():
 started=time.time();groups={};total=0
 lc=module('cuev2_list5_common',R/'experiments/list5-audio-20261111/common.py')
 for ds in ['mpd','m4a']:
  for label,roots in [('first_item',[(a,lc.old_root(ds,a)) for a in ['ddbc_joint','drop','ddbc_sid','direct','retrieved','tiger_sid']]),('five_item',[(a,lc.O/ds/a) for a in lc.ARMS])]:
   # Original full first-item arm lives in generation-final rather than the repaired list5 root.
   if label=='first_item':roots[0]=('ddbc_joint',R/'outputs/generation-final-20261107/full' if ds=='mpd' else R/'outputs/two-dataset-fair-20261108/m4a/audio/full')
   for arm,root in roots:
    files=sorted((root/'verbalization').glob('*/*.json'));hits=[];field_hits=collections.Counter()
    for p in files:
     d=json.loads(p.read_text());found={}
     for field in ['lyric_draft','music_attributes']:
      text=d.get(field,'');text=text if isinstance(text,str) else json.dumps(text)
      tokens=OFFENSIVE.findall(text)
      if tokens:found[field]=len(tokens);field_hits[field]+=1
     if found:hits.append(dict(id=p.stem,path=str(p),sha256=sha(p),hit_counts=found))
    groups[f'{ds}/{label}/{arm}']=dict(records=len(files),expected=941 if label=='first_item' else 500,hit_records=len(hits),field_hit_records=dict(field_hits),hits=hits);total+=len(files)
 atomic(O/'offensive-generated-text-scan.json',dict(elapsed_seconds=time.time()-started,total_records=total,expected_records=20292,regex=OFFENSIVE.pattern,scope='generated lyric_draft and music_attributes only, not input prompts; lexical screening not semantic adjudication',groups=groups));print('SCAN',total,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scan',action='store_true');a=p.parse_args();scan() if a.scan else main()
