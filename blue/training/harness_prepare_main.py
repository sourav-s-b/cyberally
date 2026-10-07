"""Package the reviewed plan with published clean source and trusted ML artifacts."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from blue.harness.scoring import sha256
from blue.training.harness_experiment import validate


def prepare(source, output):
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('publish clean reviewed source before packaging')
    config=json.loads(Path('ops/kaggle/harness/main32-plan.json').read_text())
    validate(config)
    source,output=Path(source),Path(output)
    for name,digest in config['input_hashes'].items():
        if sha256(source/name)!=digest:
            raise ValueError('artifact drift: '+name)
    if sha256('ops/kaggle/harness/requirements-legacy-lock.txt')!=config['runtime_requirements_sha256']:
        raise ValueError('runtime lock drift')
    config['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    output.mkdir(parents=True,exist_ok=True)
    for name in config['input_hashes']:shutil.copy2(source/name,output/name)
    (output/'pilot.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'dataset-metadata.json').write_text(json.dumps({
        'id':'souravsreekumar02/gpt-blue-harness-main-20261007',
        'title':'GPT Blue Harness Main Frozen 20261007',
        'licenses':[{'name':'other'}]},indent=2)+'\n')
    print('Prepared frozen cohort source',config['source_commit'],flush=True)
    return config


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='blue/results/gpt_harness/legacy_candidate')
    p.add_argument('--output',default='blue/results/gpt_harness/kaggle_main_artifacts')
    a=p.parse_args();prepare(a.source,a.output)
