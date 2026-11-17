"""Finite text-only followup on a freed reserved GPU; never renders audio."""
import os,json,time,subprocess
from pathlib import Path
R=Path('/storage_fast/ttu/GenPlaylist');O=R/'outputs/cue-v2-followup-20261113';E=R/'experiments';py=R/'envs/main/bin/python'
env=dict(os.environ,CUE_FOLLOWUP_USE_GPU='1',HF_HUB_OFFLINE='1')
steps=['cue-v2-followup-20261113/independent_content.py','cue-v2-mapping-audit-20261113/repair_diagnostics.py','cue-v2-followup-20261113/catalog_mmr.py'];records=[]
for step in steps:
 t=time.time();name=Path(step).stem
 with (O/(name+'-gpu.log')).open('a') as f:
  rc=subprocess.call([str(py),str(E/step)],env=env,stdout=f,stderr=subprocess.STDOUT)
 records.append(dict(step=step,started_epoch=t,elapsed_seconds=time.time()-t,returncode=rc))
 (O/'gpu-finish-status.json').write_text(json.dumps(records,indent=2)+'\n')
 if rc:raise SystemExit(rc)
print('FINITE_TEXT_FOLLOWUP_COMPLETE',flush=True)
