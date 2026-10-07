"""Local bounded MAPPO/A2C pipeline check. No ranking or success claim."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from blue.harness.scoring import sha256
from blue.training.harness_rl import check_seeds, source_identity


def run(args):
    check_seeds(args.eval_seeds)
    out=Path(args.out).resolve()
    if out.exists():raise FileExistsError('new output directory required')
    out.mkdir(parents=True)
    scorer=str(Path(args.scorer).resolve())
    plan={'source_commit':source_identity(),'scorer_sha256':sha256(scorer),
          'algorithms':['mappo','a2c'],'training_rng_seeds':[0,1],
          'training_episode_seeds':[7706,7707,7708,7709],
          'iters':2,'eps_per_iter':2,'steps':400,'threads':1,
          'ml_inputs':'risk','eval_seeds':args.eval_seeds,'policy_replicas':2,
          'interpretation':'reused development pipeline check; not enough training or independent replication to rank algorithms'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env=dict(os.environ)
    env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONHASHSEED='0')

    def job(algorithm,seed):
        tag=f'{algorithm}_s{seed}'
        common=[sys.executable,'-m','blue.training.harness_redesign']
        with (out/(tag+'.log')).open('w') as log:
            subprocess.run(common+['train','--algorithm',algorithm,'--out',str(out/tag),
                '--scorer',scorer,'--seed',str(seed),'--iters','2','--eps-per-iter','2',
                '--steps','400','--threads','1','--ml-inputs','risk'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run(common+['eval','--model-dir',str(out/tag),'--out',str(out/(tag+'_eval')),
                '--scorer',scorer,'--steps','400','--threads','1','--eval-seeds',*[str(s) for s in args.eval_seeds],
                '--policy-replicas','2'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((out/(tag+'_eval')/'report.json').read_text())
        print(json.dumps({'model':tag,'sample_returns':{s:[r['return'] for r in rows]
              for s,rows in report['cells']['sample'].items()}}),flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(job,a,s) for s in plan['training_rng_seeds'] for a in plan['algorithms']]
        for future in futures:future.result()
    (out/'complete.json').write_text(json.dumps({'status':'complete','models':4,'claim':plan['interpretation']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scorer',default='blue/results/gpt_harness/legacy_candidate/scorer.pkl')
    p.add_argument('--out',required=True);p.add_argument('--eval-seeds',type=int,nargs='+',default=[8233])
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    run(p.parse_args())
