"""Author-authorized MMR generation extension; original three arms remain unchanged."""
from build import legacy_mmr
from common import *
from gate import prepare
from sentence_transformers import SentenceTransformer
model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cuda')
model.max_seq_length=256
for ds in ['mpd','m4a']:
 legacy_mmr(ds,model,gate=True)
 check=json.loads((O/ds/'v1-gate-reconstruction-check.json').read_text())
 assert check['same_top8_set_rows']==check['rows'], (ds,check['same_top8_set_rows'],check['rows'])
 # Original v1 ordering is preserved in the control. Order-only differences
 # within the same eight cues are recorded, never silently changed.
prepare(only_arm='mmr')
