"""Lossless surface-form expansion of one-word candidates; same evidence filters."""
import sys,re,json,time,collections
from pathlib import Path
sys.path.insert(0,'/storage_fast/ttu/GenPlaylist/experiments/cue-v2-repair-r3-20261113')
from pilot_extract import O,postfilter,evidence_units,paths,atomic,normalized,sha
D=O.parent/'cue-v2-repair-r3-format-20261113'
NOUNS={'love','anger','fear','joy','hope','grief','calm','sorrow','pride','longing','nostalgia','regret','desire','despair','anxiety','envy','awe','relief','sadness','tenderness','vulnerability','uncertainty','resignation','affection'}
def expand(phrase,cat):
 if len(re.findall(r'[a-z]+',phrase))!=1 or not re.fullmatch(r'[a-z]+',phrase):return phrase
 if cat=='sound':return phrase+' music'
 if cat=='emotion':return ('expressed '+phrase) if phrase in NOUNS or phrase.endswith(('ness','tion','sion','ment')) else phrase+' feeling'
 return 'themes of '+phrase

def main():
 cfg=json.loads((O/'pilot-config.json').read_text());report={};allnorm=[]
 for ds,ids in cfg['selected'].items():
  meta=json.loads((paths(ds)['data']/'catalog_metadata.json').read_text());details=[];capacity=0;eligible=0
  for i in ids:
   old=json.loads((O/ds/'candidates'/f'{i}.json').read_text());parsed=json.loads(json.loads((O/ds/'raw-responses'/f'{i}.json').read_text())['raw']);changes=[]
   for cat,rows in parsed.items():
    for c in rows:
     p=normalized(c['phrase']);new=expand(p,cat)
     if new!=p and p!=normalized(meta[i].get('title','')):changes.append(dict(old=p,new=new,category=cat));c['phrase']=new
   accepted,rejected=postfilter(parsed,old['input'],meta[i],old['evidence_units']);unique={c['phrase'] for c in accepted};n=len(unique)
   if old['input']['lyrics']:eligible+=1;capacity+=min(8,n)
   atomic(D/ds/'candidates'/f'{i}.json',dict(old,input=old['input'],accepted=accepted,rejected=rejected,format_normalizations=changes,parent_candidate_sha256=sha(O/ds/'candidates'/f'{i}.json')));details.append(dict(item=i,old_accepted=len(old['accepted']),new_unique=n));allnorm.extend(changes)
  report[ds]=dict(usable_lyrics_songs=eligible,capacity_slots=capacity,candidate_capacity=capacity/(8*eligible),details=details)
 atomic(D/'report.json',dict(datasets=report,format_changes=allnorm,original_candidates_unchanged=True,confidence_threshold=.75,not_final_mapping_coverage=True));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
