import unittest, numpy as np
from audit import cluster
class MappingTests(unittest.TestCase):
 def test_seed_to_medoid_can_drop_member_without_radius_guard(self):
  degrees=[0,-34,20,25,30,34];v=np.array([[np.cos(np.deg2rad(a)),np.sin(np.deg2rad(a))] for a in degrees]);keys=[('theme',str(i)) for i in range(len(v))];support={k:set(range(20 if j==0 else 5)) for j,k in enumerate(keys)}
  old=cluster(keys,v,support,.82);fixed=cluster(keys,v,support,.82,.78)
  self.assertEqual(len(old),1);self.assertLess(old[0]['min_member_cosine'],.78)
  self.assertGreaterEqual(fixed[0]['min_member_cosine'],.78)
 def test_categories_do_not_mix(self):
  v=np.array([[1.,0.],[1.,0.]]);keys=[('sound','soft rock'),('theme','soft rock')];support={k:{1,2} for k in keys}
  self.assertEqual(len(cluster(keys,v,support,.82,.78)),2)
if __name__=='__main__':unittest.main()
