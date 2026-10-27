"""Private Kaggle pilot; frozen source/artifact input, fresh PPO, all runs retained."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
inputs = list(Path('/kaggle/input').rglob('pilot.json'))
if len(inputs) != 1:
    raise RuntimeError('attach exactly one frozen harness pilot dataset')
data = inputs[0].parent
cfg = json.loads(inputs[0].read_text())
for name, digest in cfg['input_hashes'].items():
    if hashlib.sha256((data / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('input hash mismatch: ' + name)
# Only experiment artifacts belong in working: Kaggle exports every file there.
# Exporting Torch headers and the full venv caused slow/rate-limited retrieval.
root = Path('/tmp/gpt-blue-repo')
# Public code only; the private dataset holds weights and frozen configuration.
subprocess.run(['git', 'clone', '--no-checkout', cfg['source_repository'], str(root)], check=True)
subprocess.run(['git', '-C', str(root), 'checkout', '--detach', cfg['source_commit']], check=True)
actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
if actual != cfg['source_commit']:
    raise RuntimeError('source commit mismatch')
# Isolate Python as well as dependencies; Kaggle's default runtime is not our pin.
subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv==0.12.15'], check=True)
profile = cfg.get('runtime_profile', 'pilot')
profiles = {'pilot': ('3.12', 'requirements.txt'),
            'legacy-candidate': ('3.11', 'requirements-legacy-lock.txt')}
if profile not in profiles:
    raise RuntimeError('unknown frozen runtime profile')
python_version, requirements = profiles[profile]
python_version = cfg.get('python_version', python_version)
requirements_path = root / 'ops/kaggle/harness' / requirements
if cfg.get('runtime_requirements_sha256') and hashlib.sha256(requirements_path.read_bytes()).hexdigest() != cfg['runtime_requirements_sha256']:
    raise RuntimeError('runtime requirements hash mismatch')
subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', python_version, '/tmp/gpt-blue-env'], check=True)
python = '/tmp/gpt-blue-env/bin/python'
subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', python,
                '--torch-backend', 'cpu', '-r', str(root / 'ops/kaggle/harness' / requirements)], check=True)
os.chdir(root)
sys.path.insert(0, str(root))
# Validate golden model predictions and a live simulator episode before training.
subprocess.run([python, '-m', 'blue.training.harness_preflight', '--input', str(data)], check=True)
output = Path('/kaggle/working/pilot')
output.mkdir(exist_ok=True)
if cfg.get('experiment') == 'main32':
    # Keep parent bootstrap independent of simulator/ML dependencies in child venv.
    seed = int(os.environ['GPT_BLUE_SHARD_SEED'])
    if seed not in cfg['training_rng_seeds']:
        raise RuntimeError('unknown frozen training shard')
    cohort_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    cfg = {**cfg, 'cohort_training_rng_seeds': cfg['training_rng_seeds'],
           'training_rng_seeds': [seed], 'cohort_sha256': cohort_hash}
(output / 'pilot.json').write_text(json.dumps(cfg, indent=2))
print(f"EXPERIMENT START source={cfg['source_commit']} runtime={profile} "
      f"training_seeds={cfg['training_rng_seeds']} cohort={cfg.get('cohort_sha256', 'pilot')}", flush=True)
for seed in cfg['training_rng_seeds']:
    for arm in cfg['arms']:
        print(f"TRAIN START arm={arm} seed={seed} iterations={cfg['iters']}", flush=True)
        dest = output / f'{arm}_s{seed}'
        cmd = [python, '-m', 'blue.training.harness_rl', '--scorer', str(data/'scorer.pkl'),
               '--out', str(dest), '--seed', str(seed), '--ml-inputs', arm,
               '--iters', str(cfg['iters']), '--eps-per-iter', str(cfg['eps_per_iter']),
               '--steps', str(cfg['steps']), '--max-age', str(cfg['max_age']),
               '--hidden', str(cfg['hidden']), '--lr', str(cfg['lr']),
               '--bonus', str(cfg['bonus']), '--temp', str(cfg['temp']),
               '--train-seeds', *map(str, cfg['train_episode_seeds'])]
        if (dest / 'checkpoint.pt').exists():
            cmd.append('--resume')
        if cfg.get('diagnostics', False):
            cmd.append('--diagnostics')
        subprocess.run(cmd, check=True)
subprocess.run([python, '-m', 'blue.training.harness_pilot', '--config', str(output/'pilot.json'), '--output', str(output), '--scorer', str(data/'scorer.pkl')], check=True)
print(f"SHARD COMPLETE: {len(cfg['arms'])*len(cfg['training_rng_seeds'])} final models evaluated; see pilot/report.json", flush=True)
