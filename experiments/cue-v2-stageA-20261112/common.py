"""Stage A only. Never mutate v1 inputs or start training."""
import hashlib, importlib.util, json, os, re, sys, time
from pathlib import Path
R=Path('/storage_fast/ttu/GenPlaylist')
E=R/'experiments/cue-v2-stageA-20261112'
O=R/'outputs/cue-v2-stageA-20261112'
REV='cdbee75f17c01a7cc42f958dc650907174af0554'
MODEL=R/'cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots'/REV
CATEGORIES=['sound','emotion','theme']
OFFENSIVE=re.compile(r'\b(?:nigg(?:a|as|er|ers)|fagg?(?:ot|ots)|cunts?|motherfuck\w*|fuck\w*|bitch(?:es)?|sluts?|whores?)\b',re.I)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def immutable(p,x):
 if Path(p).exists():assert json.loads(Path(p).read_text())==x,str(p)
 else:atomic(p,x)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def paths(ds):return module('cuev2_source_paths',R/'experiments/history-cue-baselines-20261110/common.py').paths(ds)
def normalized(s):return re.sub(r'\s+',' ',str(s).lower()).strip()
def lyric_quality(text):
 lines=[l.strip() for l in text.splitlines() if l.strip()]; words=re.findall(r"[A-Za-z']+",text)
 if len(words)<30:return 'missing_or_short'
 listing=sum(bool(re.search(r'(?:\bfeat\.?\b|\bft\.\s|\btrack\s*\d|\d{1,2}:\d{2})',l,re.I)) for l in lines)
 if len(lines)>=10 and listing/len(lines)>.35:return 'suspected_listing'
 if re.search(r'upcoming releases|release calendar|track listing|tracklist:',text,re.I):return 'suspected_listing'
 # Tokenized/stemmed lyrics remain usable for thematic evidence, but are explicitly flagged.
 return 'tokenized_text' if len(lines)<=2 and len(words)>80 else 'usable_lines'
def lyric_path(ds,i,meta):
 options=[Path(meta.get('lyrics_path') or '/nonexistent')]
 if ds=='mpd':options += [Path('/storage/ttu/GenPlaylist/restored/mpd-backup-20260615/data/lyrics/spotify')/(i+'.txt'),Path('/storage/ttu/GenPlaylist/runtime/mpd/data/lyrics/spotify')/(i+'.txt'),R/'original-ncl/data/lyrics/spotify'/(i+'.txt'),Path('/storage/ttu/GenPlaylist/data/lyrics/spotify')/(i+'.txt')]
 else:options += [R/'original-ncl/data/catalog-music4all-onion-v3-u500/lyrics'/(i+'.txt')]
 return next((p for p in options if p.is_file()),None)
def evidence_input(ds,i,meta):
 p=lyric_path(ds,i,meta);raw=p.read_text(errors='replace') if p else meta.get('lyric_excerpt','')
 quality=lyric_quality(raw)
 lyrics='\n'.join(dict.fromkeys(l.strip() for l in raw.splitlines() if l.strip()))[:6000] if quality not in ['missing_or_short','suspected_listing'] else ''
 artist=normalized(meta.get('artist',''));title=normalized(meta.get('title',''))
 tags=[t for t in meta.get('tags',[]) if normalized(t) not in {artist,title} and not OFFENSIVE.search(t)]
 return dict(item_id=i,title=meta.get('title',''),genre=meta.get('genre',''),mood=meta.get('mood',''),tags=tags,lyrics=lyrics,lyric_quality=quality,lyrics_file=None if p is None else str(p),lyrics_sha256=None if p is None else sha(p))
