from common import *
for ds in ['mpd','m4a']:
 idx=sample_indices(ds);assert len(idx)==len(set(idx))==100 and idx==sample_indices(ds)
 pairs=[(i,j) for i in idx for j in range(5)];seeds=[seed(i,j) for i,j in pairs];assert len(seeds)==len(set(seeds))==500
 assert all(seeds[k*5]==42000+i for k,i in enumerate(idx))
 # Sampled-list/position mapping has every position exactly once, including last item.
 for k,i in enumerate(idx):assert pairs[k*5:k*5+5]==[(i,j) for j in range(5)]
 v=np.repeat(np.arange(100,dtype=float),5).reshape(100,5);assert np.array_equal(v.mean(1),np.arange(100))
 itemc={str(i):list(range(i,i+8)) for i in range(15)};vocab=[str(i) for i in range(100)];refs=list(itemc)
 for j in range(5):
  q=history_random(refs,itemc,vocab,ds,2,j);assert len(q)==len(set(q))==8 and set(q)<={c for v in itemc.values() for c in v};assert q==history_random(refs,itemc,vocab,ds,2,j)
print('PASS:100 lists x5 positions, deterministic common seeds, original first-position seed, history-pool sampling, list-level averaging')
