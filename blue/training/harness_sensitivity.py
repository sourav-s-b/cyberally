"""Bounded local one-factor study; reused development, no superiority claim."""
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
from blue.training import harness_redesign as hr
from blue.training.harness_rl import check_seeds, source_identity
from blue.harness.scoring import sha256

OUT=Path('blue/results/gpt_harness/v2_sensitivity')
SCORER=Path('blue/results/gpt_harness/legacy_candidate/scorer.pkl').resolve()
ARMS={'reference':(.99,.2),'undiscounted':(1.,.2),'more_exploration':(.99,.5)}
SEEDS=[8233,8236,8241,8229]


def save(path, data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


def run():
    torch.set_num_threads(1)
    check_seeds(SEEDS)
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'source_commit':source_identity(),'runner_sha256':sha256(__file__),
          'scorer_sha256':sha256(SCORER),'runtime':hr.runtime(),'arms':ARMS,
          'training_rng':[0,1],'train_episode_seeds':[7706,7707,7708,7709],
          'iterations':4,'episodes_per_iteration':2,'steps':400,
          'evaluation_seeds':SEEDS,'policy_replicas':2,'algorithm':'mappo',
          'primary':'native return gain over matching zero-residual mixture',
          'claim':'reused development sensitivity; two training seeds, no superiority claim'}
    frozen=OUT/'plan.json'
    if frozen.exists() and json.loads(frozen.read_text())!=json.loads(json.dumps(plan)):
        raise ValueError('frozen study drift')
    save(frozen,plan)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONHASHSEED='0')

    def train(arm,seed):
        gamma,explore=ARMS[arm]; model=OUT/f'{arm}_s{seed}'
        manifest=model/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()).get('status')=='complete':return
        command=[sys.executable,'-m','blue.training.harness_redesign','train',
                 '--algorithm','mappo','--scorer',str(SCORER),'--out',str(model),
                 '--seed',str(seed),'--iters','4','--eps-per-iter','2','--steps','400',
                 '--threads','1','--ml-inputs','risk','--gamma',str(gamma),'--explore',str(explore)]
        if (model/'checkpoint.pt').exists():command+=['--resume']
        with (OUT/f'{arm}_s{seed}.log').open('a') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'trained {arm} seed {seed}',flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda job:train(*job),[(a,s) for s in [0,1] for a in ARMS]))
    cells_path=OUT/'cells.json';cells=json.loads(cells_path.read_text()) if cells_path.exists() else {}
    for arm in ARMS:
        for seed in [0,1]:
            for episode in SEEDS:
                for replica in range(2):
                    rng=episode*997+replica
                    for control in [True,False]:
                        # Discounting does not change the untrained behavior; share that control.
                        key=f'control_e{ARMS[arm][1]}_{episode}_{replica}' if control else f'{arm}_s{seed}_{episode}_{replica}'
                        if key in cells:continue
                        policy,manifest=hr.load(OUT/f'{arm}_s{seed}',SCORER,rng)
                        if control:
                            with torch.no_grad():
                                for parameter in policy.hook.actor.parameters():parameter.zero_()
                        result=hr.evaluation_episode(policy,episode,400)
                        cells[key]={**result,'policy_seed':rng,'requests':policy.request_digest.result()}
                        save(cells_path,cells)
                        print(f'{key}: {result["return"]}',flush=True)
    analysis={}
    for arm in ARMS:
        pairs=[]
        for seed in [0,1]:
            differences=[]
            for episode in SEEDS:
                differences.append(sum(cells[f'{arm}_s{seed}_{episode}_{r}']['return']-
                    cells[f'control_e{ARMS[arm][1]}_{episode}_{r}']['return'] for r in range(2))/2)
            pairs.append({'training_seed':seed,'per_episode_gain':dict(zip(SEEDS,differences)),
                          'mean_gain':sum(differences)/len(differences)})
        analysis[arm]=pairs
    save(OUT/'analysis.json',{'claim':plan['claim'],'independent_training_seeds':2,'arms':analysis})
    save(OUT/'complete.json',{'status':'complete'})

if __name__=='__main__':run()
