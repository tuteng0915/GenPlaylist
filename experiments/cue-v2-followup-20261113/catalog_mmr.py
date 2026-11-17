"""Full catalogue v1 Top8 versus same-vocabulary MMR8; no training/rendering."""
from common_followup import *
import collections
started=time.time();model=model_cpu('sentence-transformers/all-MiniLM-L6-v2');model.max_seq_length=256

def concentration(mapping,vocab,slots=8):
 occ=collections.Counter(j for row in mapping.values() for j in row[:slots] if j);song=collections.Counter(j for row in mapping.values() for j in set(row[:slots]) if j);prob=np.asarray(list(occ.values()),float)/sum(occ.values());top=occ.most_common(10);maxid,maxn=song.most_common(1)[0]
 return {'songs':len(mapping),'slots':sum(occ.values()),'max_song_fraction':maxn/len(mapping),'max_cue_id':maxid,'max_cue':vocab[maxid],'top10_occurrence_share':sum(n for j,n in top)/sum(occ.values()),'entropy_bits':float(-(prob*np.log2(prob)).sum()),'top10':[{'id':j,'cue':vocab[j],'occurrences':n,'song_fraction':song[j]/len(mapping)} for j,n in top],'pop_rap_song_fraction':sum(1 for row in mapping.values() if any(vocab[j].lower()=='pop rap' for j in row[:slots]))/len(mapping)}
for ds in ['mpd','m4a']:
 if (O/ds/'catalog-concentration.json').exists() and (O/ds/'catalog-mmr-item2cues.json').exists():continue
 t=time.time();x=paths(ds);meta=json.loads((x['data']/'catalog_metadata.json').read_text());vocab=json.loads((x['cues']/'cue_vocab.json').read_text());v1=json.loads((x['cues']/'item2cues.json').read_text());ids=sorted(v1);assert set(ids)<=set(meta);ve=unit(model.encode(vocab[1:],batch_size=64,convert_to_numpy=True));texts=[];sourcehash={}
 for i in ids:
  m=meta[i];lp=lyric_path(ds,i,m);raw=lp.read_text(errors='replace') if lp else '';sourcehash[i]=sha(lp) if lp else None;seen=set();unique=[]
  for line in raw.splitlines():
   k=normalized(line)
   if k and k not in seen:unique.append(line.strip());seen.add(k)
  lyrics='\n'.join(unique)[:2000];texts.append(' '.join(p for p in [m.get('title'),m.get('genre'),m.get('mood'),m.get('lyric_excerpt'),*m.get('tags',[])] if p)+(' '+lyrics if lyrics else ''))
 emb=unit(model.encode(texts,batch_size=64,convert_to_numpy=True));mmr={};exact=sets=0;bad=[]
 for i,v in zip(ids,emb):
  rel=ve@v;order=np.lexsort((np.arange(len(rel)),-rel));rec=(order[:8]+1).tolist();exact+=rec==v1[i][:8];sets+=set(rec)==set(v1[i][:8]);
  if set(rec)!=set(v1[i][:8]):bad.append({'item_id':i,'reconstructed':rec,'original':v1[i][:8]})
  remaining=order[:64].tolist();chosen=[]
  for _ in range(8):
   j=max(remaining,key=lambda j:(.7*float(rel[j])-.3*max([float(ve[j]@ve[k]) for k in chosen] or [0.]),-j));chosen.append(j);remaining.remove(j)
  mmr[i]=[j+1 for j in chosen]
 save(ds+'/catalog-mmr-item2cues.json',mmr)
 save(ds+'/catalog-concentration.json',{'dataset':ds,'v1':concentration(v1,vocab),'mmr':concentration(mmr,vocab),'v1_top16_context_only':concentration(v1,vocab,16),'reproduction':{'exact_order':exact,'same_top8_set':sets,'total':len(ids),'set_mismatches':bad},'protocol':{'slots':8,'candidate_pool':64,'lambda':.7,'inputs':'exact v1 title/metadata/raw lyrics deduplicated,2000chars; MiniLM256tokens','v1_source_sha256':sha(x['cues']/'item2cues.json'),'vocabulary_sha256':sha(x['cues']/'cue_vocab.json')},'lyrics_hashes':sourcehash,'elapsed_seconds':time.time()-t})
 print('CATALOG_DONE',ds,len(ids),'same_set',sets,flush=True)
print('TOTAL_SECONDS',time.time()-started,flush=True)
