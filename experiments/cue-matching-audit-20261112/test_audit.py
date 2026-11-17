import unittest
from audit import compare
class CueMatchingTests(unittest.TestCase):
    def test_identical_and_permutation(self):
        x=[[i]*8 for i in range(5)]
        self.assertEqual(compare(x,x[::-1]),(1.,1.))
    def test_pooling_hides_item_mismatch(self):
        x=[[i]*8 for i in range(5)];y=[[(i*8+j)%5 for j in range(8)] for i in range(5)]
        pooled,matched=compare(x,y);self.assertEqual(pooled,1.);self.assertLess(matched,.3)
    def test_duplicate_not_unlimited_credit(self):
        x=[[0]*8]*5;y=[list(range(8))]*5
        self.assertEqual(compare(x,y),(.125,.125))
if __name__=='__main__':unittest.main()
