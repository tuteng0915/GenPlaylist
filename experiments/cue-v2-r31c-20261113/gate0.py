"""Recompute the author's r3.1 gate, retaining all failures and deterministic examples."""
import collections,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from run import O,S,OLD,sha,atomic,inspect,old

def main():
    result={};rewrites=[];hashes={}
    for ds,threshold in [('mpd',.85),('m4a',.70)]:
        ids=json.loads((S/ds/'cohort.json').read_text())['pilot'];base=O/'pilot'/ds
        assert (base/'complete-0.json').exists(),ds
        meta=json.loads((old.paths(ds)['data']/'catalog_metadata.json').read_text());names={old.norm(m.get('artist','')) for m in meta.values() if m.get('artist')}
        counts=collections.Counter();reasons=collections.Counter();rows=[]
        for i in ids:
            p=base/'candidates'/f'{i}.json';d=json.loads(p.read_text());assert d['config_sha256']==sha(base/'config.json');hashes[f'{ds}/{i}']=sha(p)
            assert d['input']==json.loads((OLD/ds/'candidates'/f'{i}.json').read_text())['input']
            for c in d['accepted']:
                assert not old.forbidden(c['phrase'],meta[i],names)
                if c['source']=='lyrics':assert inspect(c,d['input'],meta[i],names,ds)[1] is None,(ds,i,c)
            nt=sum(c['category']=='theme' for c in d['accepted']);lyric=bool(d['input']['lyrics'])
            counts['songs']+=1;counts['lyric_songs']+=lyric;counts['lyric_theme_ge4']+=lyric and nt>=4
            counts['accepted_candidates']+=len(d['accepted']);counts['capacity_slots']+=min(8,len(d['accepted']))
            if lyric:counts['lyric_capacity_slots']+=min(8,len(d['accepted']))
            for r in d['rewrites']:
                rewrites.append(dict(dataset=ds,item_id=i,**r));counts['rewritten']+=1;counts['rewrite_remaining_excluded']+=r['remaining_exclusion'] is not None
            reasons.update(c['reason'] for c in d['rejected'] if c.get('category')=='theme')
            rows.append(dict(item_id=i,has_lyrics=lyric,theme_candidates=nt,accepted_candidates=len(d['accepted'])))
        ratio=counts['lyric_theme_ge4']/counts['lyric_songs']
        result[ds]=dict(counts=dict(counts),theme_ge4_fraction=ratio,required_fraction=threshold,passed=ratio>=threshold,remaining_theme_rejections=dict(reasons),precanonical_candidate_capacity=counts['lyric_capacity_slots']/(8*counts['lyric_songs']),songs=rows)
    examples=random.Random(42).sample(rewrites,min(10,len(rewrites)))
    report=dict(version='r3.1',datasets=result,passed=all(d['passed'] for d in result.values()),random_rewrite_examples=examples,example_sampling='uniform over every attempted rewrite, seed42; not only successes',candidate_hashes=hashes,scope='fixed200 songs; candidate support gate, NOT postmapping gate1; no-lyric minimum waived by author',full_extraction_authorized_on_pass=True,gate1='postmapping lyrical slot coverage>=.85; MRR>=v1; concentration<MMR; report before training if any failure')
    atomic(O/'gate0.json',report);print(json.dumps({k:v for k,v in report.items() if k!='candidate_hashes'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
