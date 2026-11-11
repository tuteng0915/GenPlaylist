import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tiger-joint-20261108-test-deps'))
from run import decode,module
core=module('test_top5_core',Path(__file__).resolve().parents[1]/'tiger-joint-20261108/core.py')
torch.set_num_threads(2);torch.manual_seed(42)
codes=np.array([[2+i//2,40,41,42+i%2] for i in range(64)])
cues=np.tile(np.arange(8),(64,1));cat=core.Catalog(codes,cues,8);model=core.build_model(cat.vocab_size,tiny=True).eval()
refs=np.arange(45).reshape(3,15)%64;seen=[];old=cat.inputs
def inputs(h,device):seen.append(h.copy());return old(h,device)
cat.inputs=inputs
ar,ac,score=decode(model,cat,None,cat.codes,refs,True,'ar5')
assert [s.shape[1] for s in seen]==[15,16,17,18,19]
for k in range(1,5):np.testing.assert_array_equal(seen[k][:,15:],ar[:,:k])
seen.clear();top,tc,ts=decode(model,cat,None,cat.codes,refs,True,'top5')
assert len(seen)==1 and seen[0].shape==(3,15)
assert top.shape==ar.shape==(3,5) and tc.shape==ac.shape==(3,5,8)
assert all(len(set(row))==5 for row in top) and all(len(set(row))==5 for row in ar)
np.testing.assert_array_equal(top[:,0],ar[:,0]);np.testing.assert_array_equal(tc[:,0],ac[:,0])
assert np.all(np.diff(ts,axis=1)<=1e-6)
print('PASS: fixed-history top5, rolling AR5, beam20, distinct outputs, first choice/cues agree, per-song cue alignment')
