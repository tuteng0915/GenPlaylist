import argparse,subprocess
from common import *
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--arm',required=True);p.add_argument('--metric',choices=['mert','clap','fad'],required=True);a=p.parse_args();b=O/a.dataset/a.arm;rows=json.loads((b/'render-plan.json').read_text())['rows'];system=SYSTEMS[a.arm];x=paths(a.dataset)
assert len(rows)==500
# Always validate all five positions before metric extraction.
subprocess.run([str(R/'envs/ace-step/bin/python'),str(E/'validate.py'),'--dataset',a.dataset,'--arm',a.arm,'--start','0','--count','500'],check=True)
if a.metric=='clap':
 m=module('list5_clap',CODE/'scripts/evaluate_end_to_end_clap.py');m.EXPECTED_EXAMPLES=500;old_ci=m._bootstrap_mean_interval
 m._bootstrap_mean_interval=lambda values,**kw:old_ci(np.asarray(values).reshape(100,5).mean(1),**kw)
 out=b/'metrics/clap';slug=m._slug(system);end=out/f'{slug}_clap.json'
 if end.exists():print('CLAP complete');sys.exit(0)
 ck=Path(json.loads((R/'experiments/generation-final-20261107/eval-assets.json').read_text())['lukewys/laion_clap']['path'])
 sys.argv=['clap','--audio-dir',str(b/'synthesis'),'--verbalization-dir',str(b/'verbalization'),'--checkpoint',str(ck),'--output-dir',str(out),'--system',system,'--seed-base','27000'];m.main()
 q=json.loads(end.read_text());q.update(lists=100,positions_per_list=5,confidence_interval_unit='mean of5 item scores per list; bootstrap100 lists',seed_rule='27000 + flat_index; common crop seed across methods');atomic(end,q)
elif a.metric=='fad':
 m=module('list5_fad',R/'experiments/generation-final-20261107/fad_arm.py');m.EXPECTED_EXAMPLES=500;m._test_targets=lambda unused:[r['target'] for r in rows]
 dest=b/'metrics/fad.json'
 if dest.exists():print('FAD complete');sys.exit(0)
 refs=O/a.dataset/'reference-audio';assert all((refs/(r['target']+'.mp3')).exists() for r in rows),'Missing reference audio; do not reduce cohort'
 sys.argv=['fad','--audio-dir',str(b/'synthesis'),'--prepared-dir',str(x['cohort']),'--catalog-audio-dir',str(refs),'--work-dir',str(O/a.dataset/'fad'/a.arm),'--output',str(dest),'--audio-load-workers','4','--system',system];m.main()
 q=json.loads(dest.read_text());q.update(lists=100,positions_per_list=5,reference_unit='all5 actual target songs per sampled list; occurrence-weighted',comparability='new500-clip corpus; do not compare numerical FAD against old941-first-item corpus');atomic(dest,q)
else:
 sys.path[:0]=[str(CODE/'scripts'),str(CODE/'src')]
 from extract_mert_embeddings import _load_center_audio,_pool_valid_frames
 import torch
 from transformers import AutoModel,Wav2Vec2FeatureExtractor
 revision='12af15fef9d0ac838c3f475bfbbf26d2060dd4f5';processor=model=None;out=b/'metrics/mert-rows';out.mkdir(parents=True,exist_ok=True)
 for row in rows:
  i=row['flat_index'];audio=b/'synthesis'/system/f'{i:04d}.mp3';meta=json.loads(audio.with_suffix('.json').read_text());dest=out/f'{i:04d}.npz';identity=dict(audio_sha256=sha(audio),model='m-a-p/MERT-v1-95M',revision=revision,flat_index=i,source_history_index=row['source_history_index'],item_position=row['item_position'],sample_rate=24000,clip_seconds=30,pooling='attention-masked final-layer mean',normalization='L2')
  if dest.exists():q=np.load(dest);assert json.loads(str(q['identity']))==identity;continue
  if model is None:
   processor=Wav2Vec2FeatureExtractor.from_pretrained(identity['model'],revision=revision,trust_remote_code=True);model=AutoModel.from_pretrained(identity['model'],revision=revision,trust_remote_code=True).eval().cuda()
  wave=_load_center_audio(audio,24000,720000);batch=processor([wave],sampling_rate=24000,padding=True,return_attention_mask=True,return_tensors='pt');batch={k:v.cuda() for k,v in batch.items()}
  with torch.inference_mode():h=model(**batch).last_hidden_state;emb=torch.nn.functional.normalize(_pool_valid_frames(model,h,batch.get('attention_mask')).float(),dim=-1)[0].cpu().numpy()
  assert emb.shape==(768,) and np.isfinite(emb).all();tmp=dest.with_suffix('.tmp')
  with tmp.open('wb') as f:np.savez(f,embedding=emb,identity=json.dumps(identity,sort_keys=True))
  os.replace(tmp,dest)
 print('MERT complete',a.dataset,a.arm)
