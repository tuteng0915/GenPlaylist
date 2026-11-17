"""Isolated adaptation trial; original MPD test labels never used for selection."""
from pathlib import Path
import argparse, json, sys, hashlib, os

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--variant', choices=['joint_stage1','control','conditional_mask','sid_stage1','sid_stage2'], required=True)
parser.add_argument('--steps', type=int, default=2000)
parser.add_argument('--mode', choices=['train','validate','preflight'], required=True)
args = parser.parse_args()
root = Path(args.root)
project = root.parent.parent
code = project/'experiments/training-tricks-20261106/code'
sys.path.insert(0, str(code/'src'))
sys.path.insert(0, str(code/'src/03_backbone_recommender'))
import main as engine
import diffusion
import torch
from omegaconf import OmegaConf
from datasets import load_from_disk
from itertools import chain
from warmstart import sync_ema_to_model

cfg = OmegaConf.load(project/'experiments/training-tricks-20261106/base-config.yaml')
sid=args.variant.startswith('sid_')
stage1=args.variant.endswith('stage1')
cfg.dataset='music4all';cfg.data.train='music4all';cfg.evaluator.dataset='music4all';cfg.evaluator.allow_duplicate_items=True
cfg.active_cue_tokens=0 if sid else 8
cfg.model.length=102 if sid else 262
cfg.optim.lr = .0003 if stage1 else .0001
# v2 r3.1e correction: keep the specified cue loss coefficient from the
# first optimization step.  The two stages retain their original learning-rate
# schedule and warm-start relation; only the accidental stage-1 coefficient of
# 1.0 is removed.
cfg.training.layer_loss_weights.warmup.enabled=False
cfg.training.layer_loss_weights.cue_weight=.1
assert args.steps == (5000 if stage1 else 1000)
run = root/'runs'/args.variant
run.mkdir(parents=True, exist_ok=True)
import fcntl
trial_lock=open(run/'run.lock','a')
fcntl.flock(trial_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not sid
prepared = project/'outputs/cue-v2-r31e-prepared-20261113/m4a/prepared'
artifacts = project/'original-ncl/data/artifacts-music4all-onion-v3-u500'
cues = project/'outputs/cue-v2-r31e-prepared-20261113/m4a/cue-input'
cfg.data_root = str(project/'original-ncl/data/dataset-music4all-onion-v3-u500')
for key, filename in [('catalog_embeddings_path','catalog_item_embeddings.npy'),
                      ('item_id_to_row_path','item_id_to_row.json'),
                      ('semantic_tokens_path','semantic_tokens.json'),
                      ('codebook_weights_path','rvq_codebook_weights.npy')]:
    cfg[key] = str(artifacts/filename)
for key, filename in [('item2cues_path','item2cues.json'), ('cue_vocab_path','cue_vocab.json'),
                      ('cue_manifest_path','cue_manifest.json')]:
    cfg[key] = str(cues/filename)
cfg.prepared_dataset_path = str(prepared)
cfg.loader.num_workers = 4
cfg.seed = 1
cfg.loader.batch_size = 128
cfg.loader.global_batch_size = 512
cfg.trainer.devices = 1
cfg.trainer.num_nodes = 1
cfg.trainer.accumulate_grad_batches = 4
cfg.trainer.max_steps = args.steps
cfg.trainer.enable_progress_bar = False
cfg.trainer.log_every_n_steps = 25
cfg.strategy = {'_target_':'lightning.pytorch.strategies.SingleDeviceStrategy','device':'cuda:0'}
cfg.training.log_grad_norm = True
cfg.eval_batch_size = 32
cfg.sampling.steps = 256
cfg.eval.allow_protocol_override = True
cfg.eval.disable_ema = False
cfg.checkpointing.save_dir = str(run)
last = run/'checkpoints/last.ckpt'
cfg.checkpointing.resume_from_ckpt = last.exists()
cfg.checkpointing.resume_ckpt_path = str(last)
cfg.checkpointing.warmstart_path = None if last.exists() else str(project/'original-ncl/checkpoints/pretrained/ddbc/spotify30.ckpt' if stage1 else root/('runs/'+('sid_stage1' if sid else 'joint_stage1')+'/checkpoints/last.ckpt'))
cfg.callbacks.checkpoint_every_n_steps.dirpath = str(run/'checkpoints')
cfg.callbacks.checkpoint_every_n_steps.every_n_train_steps = 100
cfg.callbacks.checkpoint_every_n_steps.save_top_k = 0
cfg.callbacks.checkpoint_every_n_steps.save_last = True
cfg.run_name = 'm4a-'+args.variant+'-seed1-adaptation-holdout'
cfg.optimization_trial = {'variant':args.variant,'validation':'training-playlist-holdout-v1',
                          'split_sha256':hashlib.sha256((root/'split.json').read_bytes()).hexdigest(),
                          'cue_weight_all_steps':.1}
split = json.loads((root/'split.json').read_text())
source=project/'original-ncl/data/processed/genplaylist-music4all-onion-v3-u500-8cue-k5-u8-u3-cap16'
assert hashlib.sha256((source/'prepared_manifest.json').read_bytes()).hexdigest()==split['source_manifest_sha256']
raw8=load_from_disk(str(source/'raw_dataset'))['train']
raw0=load_from_disk(str(prepared/'raw_dataset'))['train']
assert raw0['item_seq']==raw8['item_seq'] and raw0['bundle']==raw8['bundle']
engine.L.seed_everything(1)
logger = engine.utils.get_logger('training-tricks')
dataset = engine.AbstractDataset(cfg)
dataset.split()
tokenizer = engine.dataloader.get_tokenizer(cfg, dataset)
data, manifest = engine.load_prepared_tokenized_dataset(prepared, cfg, dataset, tokenizer)
raw = load_from_disk(str(prepared/'raw_dataset'))['train']
train_indices = split['training_indices']
valid_indices = split['validation_indices']
import re
groups = lambda indices: {re.search(r'm4a-train-([a-f0-9]+)-',raw[int(i)]['bundle']).group(1) for i in indices}
assert not groups(train_indices) & groups(valid_indices)
data['train'] = data['train'].select(train_indices)
# Only training-source held-out windows become this run's evaluation partition.
raw_validation = raw.select(split['pilot_validation_indices'])
data['test'] = tokenizer.tokenize({'test':raw_validation})['test']
os.chdir(run)
OmegaConf.save(cfg, run/'resolved-config.yaml', resolve=True)
preflight = {'variant':args.variant,'train_rows':len(data['train']), 'eval_rows':len(data['test']),
             'original_test_used':False,'playlist_overlap':0,'global_batch':512,
             'seed':1,'steps':args.steps,'runtime_torch':torch.__version__,
             'split_sha256':cfg.optimization_trial.split_sha256}
(run/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
if args.mode == 'preflight':
    print(json.dumps(preflight)); sys.exit(0)
if not stage1:
    def exact_warmstart(model,path):
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['global_step']==5000
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        model.ema.load_state_dict(checkpoint['ema'])
        model.ema.copy_to(chain(model.backbone.parameters(),model.noise.parameters()))
        sync_ema_to_model(model)
        return {'source':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                'start':'SID-only stage1 EMA, reset optimizer and schedule, matched 5000+1000'}
    engine.apply_ddbc_warmstart=exact_warmstart
if args.variant=='conditional_mask':
    from training_objectives import install
    install(diffusion.Diffusion,diffusion.Loss,'conditional_mask')
if args.mode == 'train':
    if last.exists():
        old = torch.load(last,map_location='cpu',weights_only=False)
        old_trial = old.get('hyper_parameters',{}).get('config',{}).get('optimization_trial',{})
        assert old_trial.get('variant') == args.variant
        assert old_trial.get('split_sha256') == cfg.optimization_trial.split_sha256
        if int(old['global_step']) >= args.steps:
            print('Already reached requested step budget; no duplicate training.'); sys.exit(0)
        del old
    engine._train(cfg,logger,tokenizer,data)
else:
    if not last.exists(): raise FileNotFoundError(last)
    step = int(torch.load(last,map_location='cpu',weights_only=False)['global_step'])
    result_path = run/f'validation-step{step}.json'
    if result_path.exists():
        print('Validation already exists; skip.'); sys.exit(0)
    cfg.eval.checkpoint_path = str(last)
    cfg.eval.results_path = str(result_path)
    engine._rec_eval(cfg,logger,tokenizer,data)
    result = json.loads(result_path.read_text())
    result['evaluation']['partition'] = 'adaptation-training-playlist-heldout-512'
    result['evaluation']['official_protocol'] = False
    result['evaluation']['selection_only'] = True
    result['evaluation']['split_sha256'] = cfg.optimization_trial.split_sha256
    result_path.write_text(json.dumps(result,indent=2)+'\n')
