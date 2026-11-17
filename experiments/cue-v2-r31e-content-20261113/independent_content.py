"""Fixed actual-lyrics and actual-metadata references; no cue vocabulary in metric."""
from common_followup import *
from metrics import text_embedding
started=time.time();model=model_cpu('sentence-transformers/all-mpnet-base-v2')
for ds in ['mpd','m4a']:
 (O/ds).mkdir(exist_ok=True)
 if (O/ds/'independent-content-report.json').exists() and (O/ds/'independent-content-per-context.npz').exists():continue
 co=json.loads((S/ds/'cohort.json').read_text());x=paths(ds);meta=json.loads((x['data']/'catalog_metadata.json').read_text());ids=sorted({i for r in co['gate_rows'] for i in r['targets']});lookup={i:j for j,i in enumerate(ids)};texts=[];mt=[];valid=[];mvalid=[];qualities={};hashes={}
 for i in ids:
  m=meta[i];lp=lyric_path(ds,i,m);raw=lp.read_text(errors='replace') if lp else '';q=lyric_quality(raw);qualities[i]=q;hashes[i]=sha(lp) if lp else None
  usable=q not in ['missing_or_short','suspected_listing'];lyrics='\n'.join(dict.fromkeys(l.strip() for l in raw.splitlines() if l.strip())) if usable else '';texts.append(lyrics);valid.append(bool(lyrics));tags=m.get('tags',[]);mtext='; '.join(str(v) for v in [m.get('genre',''),', '.join(tags),m.get('mood','')] if v);mt.append(mtext);mvalid.append(bool(mtext))
 le=text_embedding(model,texts);me=text_embedding(model,mt);targetidx=np.array([[lookup[i] for i in r['targets']] for r in co['gate_rows']]);lv=np.asarray(valid)[targetidx];mv=np.asarray(mvalid)[targetidx];allout={};result={'dataset':ds,'contexts':200,'encoder':'sentence-transformers/all-mpnet-base-v2','definitions':{'lyrics':'generated lyric vs full deduplicated actual target lyrics; all300token chunks encoded; mean cosine over available lyrics among5 targets','attributes':'generated attributes vs actual genre/tags/mood of5targets; title/artist/cues excluded','missing':'report coverage; primary paired analysis uses fixed contexts with>=1usable target; all5targets-zero-for-missing secondary also retained','lyrics_quality':'short/listing lyrics excluded; tokenized/stemmed texts retained and counted; no inference that all source lyrics are correct'},'lyric_reference_contexts':int((lv.sum(1)>0).sum()),'lyric_reference_occurrences':int(lv.sum()),'metadata_reference_occurrences':int(mv.sum()),'reference_qualities':dict(__import__('collections').Counter(qualities.values())),'reference_hashes':hashes,'arms':{},'paired':{}}
 for a in ARMS:
  base=(R/'outputs/cue-v2-r31e-gate-20261113' if a=='v2' else S);files=[base/ds/'gate'/a/'verbalization/GenPlaylist'/f'{i:04d}.json' for i in range(200)];waiting(files);rows=[json.loads(f.read_text()) for f in files];ge=text_embedding(model,[r['lyric_draft'] for r in rows]);ga=text_embedding(model,[r['music_attributes'] for r in rows]);ls=np.einsum('nd,nkd->nk',ge,le[targetidx]);ms=np.einsum('nd,nkd->nk',ga,me[targetidx]);data={}
  for name,sim,mask in [('actual_lyric_fit',ls,lv),('actual_metadata_fit',ms,mv)]:
   vals=(sim*mask).sum(1)/np.maximum(mask.sum(1),1);ok=mask.sum(1)>0;data[name]=vals;data[name+'_all5_missing_zero']=(sim*mask).mean(1);allout[name+'__valid']=ok
   result['arms'].setdefault(a,{})[name]={'mean':float(vals[ok].mean()),'contexts':int(ok.sum())}
  for k,v in data.items():allout[a+'__'+k]=v
 for l,r in [('v1','none'),('mmr','none'),('mmr','v1'),('v2','none'),('v2','v1')]:
  result['paired'][l+' minus '+r]={}
  for k in ['actual_lyric_fit','actual_metadata_fit']:
   ok=allout[k+'__valid'];delta=(allout[l+'__'+k]-allout[r+'__'+k])[ok];rng=np.random.default_rng(42);means=[]
   for _ in range(50):means.extend(delta[rng.integers(len(delta),size=(200,len(delta)))].mean(1))
   result['paired'][l+' minus '+r][k]={'delta':float(delta.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'contexts':len(delta)}
 np.savez_compressed(O/ds/'independent-content-per-context.npz',**allout);result['elapsed_seconds_since_start']=time.time()-started;save(ds+'/independent-content-report.json',result);print('CONTENT_DONE',ds,flush=True)
