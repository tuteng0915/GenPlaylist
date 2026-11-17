from build import legacy_mmr
from sentence_transformers import SentenceTransformer
model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device="cuda");model.max_seq_length=256
for ds in ["mpd","m4a"]:legacy_mmr(ds,model)
