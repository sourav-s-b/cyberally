"""Bounded full-horizon MAPPO/A2C pipeline check; reused diagnostics only."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONHASHSEED='0')
source=os.environ['GPT_BLUE_SOURCE_COMMIT']
root=Path('/tmp/gpt-blue-v2-repo')
subprocess.run(['git','clone','--no-checkout','https://github.com/sourav-s-b/cyberally.git',str(root)],check=True)
subprocess.run(['git','-C',str(root),'checkout','--detach',source],check=True)
scorers=list(Path('/kaggle/input').rglob('scorer.pkl'))
if len(scorers)!=1:raise RuntimeError('attach exactly the approved replay dataset')
scorer=scorers[0]
if hashlib.sha256(scorer.read_bytes()).hexdigest()!='ec4026a3c81c2746253f306fd7add00a07f42fe13533b856fb6b4d01b8402fcd':
    raise RuntimeError('approved scorer drift')
subprocess.run([sys.executable,'-m','pip','install','uv==0.12.15'],check=True)
subprocess.run([sys.executable,'-m','uv','venv','--python','3.11.16','/tmp/gpt-blue-v2-env'],check=True)
python='/tmp/gpt-blue-v2-env/bin/python'
subprocess.run([sys.executable,'-m','uv','pip','install','--python',python,'--torch-backend','cpu','-r',str(root/'ops/kaggle/harness/requirements-legacy-lock.txt')],check=True)
os.chdir(root)
out=Path('/kaggle/working/v2-smoke');out.mkdir()
plan={'source_commit':source,'scorer_sha256':hashlib.sha256(scorer.read_bytes()).hexdigest(),
      'algorithms':['mappo','a2c'],'training_rng_seeds':[0,1],
      'training_episode_seeds':[7706,7707,7708,7709],
      'iters':2,'eps_per_iter':2,'steps':400,'threads':1,
      'ml_inputs':'risk','eval_seeds':[8233],'policy_replicas':2,
      'interpretation':'full-horizon pipeline and policy-reproduction check; not algorithm ranking, ML value or held-out improvement claim'}
(out/'plan.json').write_text(json.dumps(plan,indent=2))


def job(algorithm,seed):
    tag=f'{algorithm}_s{seed}'
    common=[python,'-m','blue.training.harness_redesign']
    with (out/(tag+'.log')).open('w') as log:
        subprocess.run(common+['train','--algorithm',algorithm,'--out',str(out/tag),
            '--scorer',str(scorer),'--seed',str(seed),'--iters','2','--eps-per-iter','2',
            '--steps','400','--threads','1','--ml-inputs','risk'],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(common+['eval','--model-dir',str(out/tag),'--out',str(out/(tag+'_eval')),
            '--scorer',str(scorer),'--steps','400','--threads','1','--eval-seeds','8233',
            '--policy-replicas','2'],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('MODEL COMPLETE '+tag,flush=True)


jobs=[(a,s) for s in plan['training_rng_seeds'] for a in plan['algorithms']]
with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(job,a,s) for a,s in jobs]
    for future in futures:future.result()
print('V2 PIPELINE COMPLETE: four models, no performance claim',flush=True)
