import unittest
import torch
from trial import project_aux
class ProjectionTests(unittest.TestCase):
    def test_conflict_removed_without_changing_primary(self):
        s=[torch.tensor([1.,0.])];c=[torch.tensor([-2.,3.])]
        out,_,_=project_aux(s,c)
        torch.testing.assert_close(out[0],torch.tensor([0.,3.]))
        torch.testing.assert_close(s[0],torch.tensor([1.,0.]))
        self.assertGreaterEqual(float((out[0]*s[0]).sum()),0.)
    def test_aligned_unchanged(self):
        s=[torch.tensor([1.,2.])];c=[torch.tensor([3.,4.])]
        out,_,_=project_aux(s,c);torch.testing.assert_close(out[0],c[0])
    def test_zero_primary_finite(self):
        out,co,_=project_aux([torch.zeros(3)],[torch.ones(3)])
        self.assertTrue(torch.isfinite(co));torch.testing.assert_close(out[0],torch.ones(3))
if __name__=='__main__':unittest.main()
