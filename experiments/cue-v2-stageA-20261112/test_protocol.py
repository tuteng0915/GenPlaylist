import unittest
import numpy as np
from scipy.optimize import linear_sum_assignment
from extract import validate_candidate
from build import choose
from metrics import oas_one_to_five,fad,content_scores
from common import lyric_quality

class ProtocolTests(unittest.TestCase):
 def test_content_masks_are_not_scores(self):
  import io
  b=io.BytesIO()
  np.savez(b,**{'v1__content_fit_v1':np.ones(200),'v1__reference_nonempty':np.ones((200,5)), 'v2__content_fit_v1':np.zeros(200)})
  b.seek(0)
  with np.load(b) as rows:
   scores=content_scores(rows,'v1')
   self.assertEqual(set(scores),{'content_fit_v1'})
   self.assertEqual(scores['content_fit_v1'].shape,(200,))
 def test_sound_must_not_be_invented_from_lyrics(self):
  c=dict(phrase='soft piano',category='sound',source='lyrics',evidence='piano',confidence=.9)
  self.assertEqual(validate_candidate(c,dict(lyrics='piano',genre='',tags=[],mood=''),{}),'sound_without_metadata')
 def test_evidence_and_names(self):
  inp=dict(lyrics='I long for the past',genre='folk',tags=[],mood='')
  c=dict(phrase='past longing',category='theme',source='lyrics',evidence='long for the past',confidence=.9)
  self.assertIsNone(validate_candidate(c,inp,{}))
  self.assertEqual(validate_candidate(dict(c,evidence='invented'),inp,{}),'unsupported_quote')
  self.assertEqual(validate_candidate(dict(c,phrase='famous artist'),inp,dict(artist='Famous Artist')),'artist_name')
 def test_no_forced_eight(self):
  e=np.eye(3);c=[dict(cue_id=1,category='theme',mapping_score=.9),dict(cue_id=2,category='theme',mapping_score=.8)]
  self.assertEqual(len(choose(c,e)),2)
  self.assertEqual(choose([],e),[])
 def test_quality_flags(self):
  self.assertEqual(lyric_quality(''), 'missing_or_short')
  self.assertEqual(lyric_quality('release calendar '+('example '*40)), 'suspected_listing')
 def test_rectangular_oas_matches_ddbc_assignment(self):
  rng=np.random.default_rng(9);p=rng.normal(size=(7,6));t=rng.normal(size=(7,5,6));oas,sims=oas_one_to_five(p,t)
  for j in range(7):
   r,c=linear_sum_assignment(1-sims[j:j+1]);self.assertAlmostEqual(oas[j],1-(1-sims[j:j+1])[r,c].sum(),places=12)
 def test_fad_zero_and_translation(self):
  x=np.random.default_rng(5).normal(size=(80,6))
  self.assertAlmostEqual(fad((x,x)),0,places=8)
  self.assertAlmostEqual(fad((x,x+1)),6,places=8)

if __name__=='__main__':unittest.main()
