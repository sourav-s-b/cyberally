"""Private bounded replay: frozen checkpoints, no training or final seeds."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONHASHSEED='0')
files=list(Path('/kaggle/input').rglob('replay.json'))
if len(files)!=1:
    raise RuntimeError('attach exactly one frozen replay dataset')
data=files[0].parent
cfg=json.loads(files[0].read_text())
for name,digest in cfg['input_hashes'].items():
    if hashlib.sha256((data/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('input drift: '+name)
root=Path('/tmp/gpt-blue-replay-repo')
subprocess.run(['git','clone','--no-checkout',cfg['source_repository'],str(root)],check=True)
subprocess.run(['git','-C',str(root),'checkout','--detach',cfg['source_commit']],check=True)
subprocess.run([sys.executable,'-m','pip','install','uv==0.12.15'],check=True)
subprocess.run([sys.executable,'-m','uv','venv','--python','3.11.16','/tmp/gpt-blue-replay-env'],check=True)
python='/tmp/gpt-blue-replay-env/bin/python'
lock=root/'ops/kaggle/harness/requirements-legacy-lock.txt'
if hashlib.sha256(lock.read_bytes()).hexdigest()!=cfg['requirements_sha256']:
    raise RuntimeError('dependency lock drift')
subprocess.run([sys.executable,'-m','uv','pip','install','--python',python,'--torch-backend','cpu','-r',str(lock)],check=True)
os.chdir(root)
out=Path('/kaggle/working/replay')
out.mkdir()
(out/'replay.json').write_text(json.dumps(cfg,indent=2))
models=Path('/tmp/gpt-blue-replay-models')
for model in {job['model'] for job in cfg['jobs']}:
    dest=models/model
    dest.mkdir(parents=True)
    for name in ('actor.th','manifest.json'):
        shutil.copyfile(data/(model+'_'+name),dest/name)
for job in cfg['jobs']:
    command=[python,'-m','blue.training.harness_numerical_audit',
             '--model-dir',str(models/job['model']), '--scorer',str(data/'scorer.pkl'),
             '--seed',str(job['seed']),'--threads',str(job['threads']),
             '--tolerance',str(job['tolerance']),'--out',str(out/(job['tag']+'.json'))]
    print('REPLAY START '+job['tag'],flush=True)
    subprocess.run(command,check=True)
print('REPLAY COMPLETE: no training; all frozen models retained',flush=True)
