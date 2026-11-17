"""Read-only evidence check after first-section results. Produces no paper changes."""
from common_followup import *
import collections
required=[O/d/f for d in ['mpd','m4a'] for f in ['catalog-concentration.json','catalog-mmr-item2cues.json','independent-content-complete.json','v1-vs-none-immediate.json']]
waiting(required)
checks={}
for ds in ['mpd','m4a']:
 x=paths(ds);v1=json.loads((x['cues']/'item2cues.json').read_text());mmr=json.loads((O/ds/'catalog-mmr-item2cues.json').read_text());report=json.loads((O/ds/'catalog-concentration.json').read_text());assert set(v1)==set(mmr)
 for arm,mp in [('v1',v1),('mmr',mmr)]:
  assert all(len(v)>=8 and len(set(v[:8]))==8 for v in mp.values())
  counts=collections.Counter(c for row in mp.values() for c in row[:8]);song_counts=collections.Counter(c for row in mp.values() for c in set(row[:8]));p=np.array(list(counts.values()),float)/sum(counts.values());r=report[arm]
  assert abs(max(song_counts.values())/len(mp)-r['max_song_fraction'])<1e-12
  assert abs(sum(sorted(counts.values(),reverse=True)[:10])/sum(counts.values())-r['top10_occurrence_share'])<1e-12
  assert abs(float(-(p*np.log2(p)).sum())-r['entropy_bits'])<1e-12
 checks[ds]={'catalog_counts_independently_recomputed':True,'same_top8_set_reproduction':report['reproduction']['same_top8_set'],'catalog_songs':len(v1),'source_hashes':{'metadata':sha(x['data']/'catalog_metadata.json'),'v1_cues':sha(x['cues']/'item2cues.json'),'vocabulary':sha(x['cues']/'cue_vocab.json')},'content_mean_check':json.loads((O/ds/'independent-content-complete.json').read_text())['independent_mean_check_passed']}
files={str(p.relative_to(O)):sha(p) for p in required};save('followup-verification.json',dict(passed=True,checks=checks,artifacts=files,stage_B_started=False,paper_changed=False,push=False));print('FOLLOWUP_VERIFIED',flush=True)
