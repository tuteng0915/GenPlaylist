"""Frozen two-dataset matched TIGER training/inference; separate output namespace."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, datetime, fcntl, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
from core import Catalog, build_model, fingerprint, loss_parts, predict

P = argparse.ArgumentParser()
P.add_argument('--dataset', choices=['mpd', 'm4a'], required=True)
P.add_argument('--arm', choices=['sid', 'joint'], required=True)
P.add_argument('--mode', choices=['preflight', 'train', 'infer'], required=True)
P.add_argument('--partition', choices=['audio', 'remaining'], default='audio')
P.add_argument('--project', type=Path, default=Path('/storage_fast/ttu/GenPlaylist'))
a = P.parse_args()
R = a.project; E = R/'experiments/cue-v2-r31e-tiger-20261113'; O = R/'outputs/cue-v2-r31e-tiger-20261113'
run = O/a.dataset/a.arm; run.mkdir(parents=True, exist_ok=True)
lock = (run/(a.mode+'-'+a.partition+'.lock')).open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.new'); temp.write_text(json.dumps(value, indent=2)+'\n'); os.replace(temp, path)

def save(path, value):
    temp = path.with_suffix('.new'); torch.save(value, temp); os.replace(temp, path)

from datasets import load_from_disk
prepared = R/'original-ncl/data/processed'/('genplaylist-v4-8cue-20item-joint-15to5' if a.dataset=='mpd' else 'genplaylist-music4all-onion-v3-u500-8cue-k5-u8-u3-cap16')
split_path = R/'experiments'/('combined-decoding-20261107/split.json' if a.dataset=='mpd' else 'two-dataset-fair-20261108/split.json')
split = json.loads(split_path.read_text()); assert sha(prepared/'prepared_manifest.json') == split['source_manifest_sha256']
if a.dataset=='mpd': assert sha(split_path)=='8f1e1316137abb106078d06c79076f90bb09624a0ef02dff0c26b93cf9ad425b'
else: assert sha(split_path)=='2d4443d88d2e153a6e0b6d1fc72d76568cd979d1f52fd0d9a161bd9deff8f2a7'
cue_root = R/'outputs/cue-v2-r31e-20261113'; cue_dir = cue_root/a.dataset
vectors = prepared/'vectors'
ids = list(map(str, json.loads((vectors/'catalog_item_ids.json').read_text())))
manifest=json.loads((prepared/'prepared_manifest.json').read_text())
assert sha(vectors/'catalog_item_ids.json')==manifest['outputs']['vectors/catalog_item_ids.json']['sha256']
assert len(ids) == (5119 if a.dataset=='mpd' else 10950) and len(ids)==len(set(ids))
rows = {s:i for i,s in enumerate(ids)}
canonical = np.load(vectors/'catalog_semantic_tokens.npy').astype(np.int64)
item_cues = json.loads((cue_dir/'item2cues.json').read_text())
assert set(ids)==set(item_cues)
cues = np.asarray([item_cues[s][:8] for s in ids], dtype=np.int64)
assert cues.shape == (len(ids),8)
cat = Catalog(canonical, cues, len(json.loads((cue_root/'cue_vocab.json').read_text())))
raw = load_from_disk(str(prepared/'raw_dataset'))
train_indices = split['training_indices']; assert not set(train_indices)&set(split['validation_indices'])
assert len(raw['test']) == (941 if a.dataset=='mpd' else 19771)
recipe = dict(dataset=a.dataset, arm=a.arm, steps=20000, seed=1, batch_size=256,
    optimizer='Adafactor; scale_parameter=False; relative_step=False', peak_lr=.001,
    lr_constant_steps=10000, dropout=.1, loss='mean SID CE + .1 mean cue CE (joint only) + .1 EOS CE',
    encoder='15..19 item SID4+catalog-cue8, same for both arms; no target leakage',
    decoder='SID4 then cue8 then EOS' if a.arm=='joint' else 'SID4 then EOS',
    vocab_size=cat.vocab_size, cue_offset=cat.cue_offset, semantic_shift=cat.shift,
    train_rows=len(train_indices), split_sha256=sha(split_path),
    prepared_manifest_sha256=sha(prepared/'prepared_manifest.json'),
    canonical_sha256=sha(vectors/'catalog_semantic_tokens.npy'), cues_sha256=sha(cue_dir/'item2cues.json'), cue_vocab_sha256=sha(cue_root/'cue_vocab.json'), cue_repair_report_sha256=sha(cue_root/'repair-report.json'),
    protocol_sha256=sha(E/'protocol.json'), core_sha256=sha(E/'core.py'))
if (run/'recipe.json').exists(): assert json.loads((run/'recipe.json').read_text())==recipe
else: dump(run/'recipe.json', recipe)
if a.mode=='preflight':
    print(json.dumps(recipe)); sys.exit(0)
random.seed(1); np.random.seed(1); torch.manual_seed(1); torch.cuda.manual_seed_all(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
assert torch.cuda.is_available(); device=torch.device('cuda')
model=build_model(cat.vocab_size); initial=fingerprint(model); model=model.to(device)
checkpoint=run/'last.ckpt'; joint=a.arm=='joint'
if a.mode=='train':
    from transformers import Adafactor
    optimizer=Adafactor(model.parameters(),lr=.001,scale_parameter=False,relative_step=False,warmup_init=False)
    gen=torch.Generator(device='cpu').manual_seed(1); start=0; digest='00'*32; elapsed_before=0.
    if checkpoint.exists():
        c=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert c['recipe']==recipe and c['initial_sha256']==initial
        model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer']);gen.set_state(c['sampling_rng'])
        torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);random.setstate(c['python_rng']);np.random.set_state(c['numpy_rng'])
        start=c['step'];digest=c['case_digest'];elapsed_before=c['training_seconds'];del c
    if start==20000:
        dump(run/'train-complete.json',{'step':20000,'checkpoint_sha256':sha(checkpoint),'case_digest':digest,'initial_sha256':initial});sys.exit(0)
    seq=raw['train'].select(train_indices)['item_seq']
    train=torch.tensor([[rows[str(x)] for x in row] for row in seq]);assert train.shape==(len(train_indices),20)
    model.train();tic=time.monotonic();lastlog=tic
    for step in range(start+1,20001):
        lr=.001 if step<=10000 else .001*math.sqrt(10000/step)
        for g in optimizer.param_groups:g['lr']=lr
        x,mask,y,rr,ss=cat.batch(train,256,gen,joint,device)
        digest=hashlib.sha256(bytes.fromhex(digest)+rr.numpy().tobytes()+ss.numpy().tobytes()).hexdigest()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            out=model(input_ids=x,attention_mask=mask,labels=y)
            loss,parts=loss_parts(out.logits,y,joint)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%100==0:
            now=time.monotonic();print(json.dumps({'step':step,'loss':float(loss),'parts':{k:float(v) for k,v in parts.items()},'seconds_since_last_log':now-lastlog}),flush=True);lastlog=now
        if step%500==0 or step==20000:
            seconds=elapsed_before+time.monotonic()-tic
            save(checkpoint,dict(step=step,recipe=recipe,model=model.state_dict(),optimizer=optimizer.state_dict(),
                sampling_rng=gen.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),case_digest=digest,
                initial_sha256=initial,training_seconds=seconds,parameters=sum(p.numel() for p in model.parameters())))
            dump(run/'progress.json',dict(step=step,case_digest=digest,initial_sha256=initial,training_seconds=seconds))
    dump(run/'train-complete.json',dict(step=20000,case_digest=digest,initial_sha256=initial,checkpoint_sha256=sha(checkpoint)))
else:
    c=torch.load(checkpoint,map_location='cpu',weights_only=False);assert c['recipe']==recipe and c['step']==20000
    model.load_state_dict(c['model']);model.eval();del c
    audio_indices=list(range(941)) if a.dataset=='mpd' else json.loads((R/'experiments/two-dataset-fair-20261108/cohort.json').read_text())['source_test_indices']
    chosen=audio_indices if a.partition=='audio' else [i for i in range(len(raw['test'])) if i not in set(audio_indices)]
    dest=run/a.partition;dest.mkdir(exist_ok=True)
    identity={'checkpoint_sha256':sha(checkpoint),'core_sha256':sha(E/'core.py'),'indices':chosen,'beam':20,'ranked_tokens':4,'cue_decode':'greedy, unrestricted full cue vocabulary','history_allowed':True,'exclude':'previous generated songs only'}
    if (dest/'identity.json').exists():assert json.loads((dest/'identity.json').read_text())==identity
    else:dump(dest/'identity.json',identity)
    source=raw['test'].select(chosen);seq=source['item_seq']
    test=np.array([[rows[str(x)] for x in row[:15]] for row in seq]); start_time=time.monotonic()
    for start in range(0,len(test),32):
        f=dest/f'batch-{start:05d}.npz'
        if f.exists():
            b=np.load(f,allow_pickle=False);assert b['pred'].shape==(min(32,len(test)-start),5);continue
        pred,generated_cues=predict(model,cat,test[start:start+32],joint,device)
        temp=f.with_suffix('.new')
        with temp.open('wb') as handle:np.savez(handle,pred=pred,cues=generated_cues)
        os.replace(temp,f);print(a.dataset,a.arm,a.partition,start,flush=True)
    batches=[np.load(dest/f'batch-{s:05d}.npz',allow_pickle=False) for s in range(0,len(test),32)]
    pred=np.concatenate([b['pred'] for b in batches]) if batches else np.empty((0,5),dtype=np.int64)
    generated_cues=np.concatenate([b['cues'] for b in batches]) if batches else np.empty((0,5,8 if joint else 0),dtype=np.int64)
    assert pred.shape==(len(chosen),5) and all(len(set(row))==5 for row in pred)
    payload=dict(result_schema='genplaylist-wp-c-joint-15to5-eval-v4',dataset=a.dataset,arm=a.arm,source_test_indices=chosen,
        inference=identity,seconds_this_invocation=time.monotonic()-start_time,
        predictions=dict(item_ids=np.array(ids)[pred].tolist(),semantic_token_ids=canonical[pred].tolist(),
            cue_ids=generated_cues.tolist(),target_item_ids=[[str(x) for x in row[15:]] for row in seq]))
    dump(dest/'plan.json',payload);dump(dest/'complete.json',{'rows':len(chosen),'plan_sha256':sha(dest/'plan.json')})
