from common import *
import tarfile,io
import soundfile as sf
b=O/'m4a';archive=b/'reference-additional.tar';expected=set(json.loads((b/'reference-missing.json').read_text()));assert sha(archive)=='cc6f45f4a14c8154025d4b447f312998665f8af4cba1bd5742c0a234c9ed9d50'
with tarfile.open(archive) as t:
 records=json.load(t.extractfile('manifest.json'));lookup={r['name']:r for r in records};assert set(lookup)=={i+'.mp3' for i in expected};assert len(t.getnames())==len(records)+1 and set(t.getnames())==set(lookup)|{'manifest.json'}
 for name,r in lookup.items():
  dest=b/'reference-audio'/name;member=t.getmember(name);assert member.isfile() and member.size==r['bytes']
  if dest.exists():assert sha(dest)==r['sha256'];continue
  temp=dest.with_suffix('.receiving');h=hashlib.sha256()
  with t.extractfile(member) as src,temp.open('wb') as out:
   for chunk in iter(lambda:src.read(8*1024*1024),b''):h.update(chunk);out.write(chunk)
  assert h.hexdigest()==r['sha256'];os.replace(temp,dest)
co=json.loads((b/'cohort.json').read_text());allids=sorted({r['target'] for r in co['rows']});verified=[]
for item in allids:
 p=b/'reference-audio'/(item+'.mp3');info=sf.info(p);assert info.duration>=30 and info.frames>0
 verified.append(dict(item=item,sha256=sha(p),duration=info.duration,bytes=p.stat().st_size))
atomic(b/'reference-validation.json',dict(passed=True,unique_files=len(verified),target_occurrences=500,source='original user-owned NCL GenPlaylist audio catalog; additional289 source hashes verified',additional_archive_sha256=sha(archive),records=verified));print('REFERENCE PASS',len(verified),'unique /500 target occurrences')
