"""Finite catalog extraction shard. Outer guard owns GPU and actual reservation deadline."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args()
for ds in ['mpd','m4a']:
 subprocess.run([sys.executable,str(Path(__file__).with_name('run.py')),'--scope','full','--dataset',ds,'--shard',str(a.shard),'--shards','4'],check=True)
