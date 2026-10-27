"""Controlled ML fitting ablation: host/episode weights and sigmoid calibration.

Keeps HGB/features/labels unchanged. Only free-agent unconfirmed contexts are
eligible; that is still a proxy for scheduling, NOT action value or scan yield.
No test-set model selection. Novelty is retained for explicit ablation only.
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from blue.harness.features import NAMES,VERSION
from blue.harness.calibration import SigmoidCalibration,grouped_weights
from blue.harness.scoring import sha256
from blue.training.harness_ml import paths,validate_splits,metrics,episode_summary


def fit(data_path,out,random_state=0):
    from sklearn.ensemble import HistGradientBoostingClassifier,IsolationForest
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from importlib.metadata import version
    import subprocess
    data,labels,manifest=paths(data_path);meta=json.loads(manifest.read_text())
    if meta['version']!=VERSION or tuple(meta['features'])!=NAMES:raise ValueError('feature drift')
    if sha256(data)!=meta['data_sha256'] or sha256(labels)!=meta['labels_sha256']:raise ValueError('dataset drift')
    validate_splits(meta['splits'])
    with np.load(data) as z:X=z['X'];episode=z['episode_seed'];agents=z['agent'];hosts=z['host']
    with np.load(labels) as z:y=z['y']
    if X.shape!=(len(y),len(NAMES)) or not np.isin(y,[0,1]).all() or np.isinf(X).any():raise ValueError('invalid data')
    if set(np.unique(episode))!=set(sum(meta['splits'].values(),[])):raise ValueError('unassigned data episodes')
    eligible=(X[:,NAMES.index('belief_CONFIRMED')]==0)&(X[:,NAMES.index('belief_VERIFY')]==0)&(X[:,NAMES.index('busy')]==0)
    masks={k:np.isin(episode,seeds)&eligible for k,seeds in meta['splits'].items()}
    if any(len(np.unique(y[m]))!=2 for m in masks.values()):raise ValueError('split lacks both labels')
    train,cal,test=[masks[k] for k in ('train','calibration','test')]
    weights={k:grouped_weights(episode[m],agents[m],hosts[m]) for k,m in masks.items()}
    W=weights['train'][:,None];observed=np.isfinite(X[train])
    den=(W*observed).sum(0)
    mean=np.divide((W*np.nan_to_num(X[train],nan=0.)).sum(0),den,out=np.zeros(len(NAMES)),where=den>0)
    centered=X[train]-mean
    scale=np.sqrt(np.divide((W*np.nan_to_num(centered**2,nan=0.)).sum(0),den,out=np.ones(len(NAMES)),where=den>0));scale=np.where(scale>1e-6,scale,1.)
    params=dict(max_iter=80,max_leaf_nodes=15,min_samples_leaf=40,learning_rate=.08,l2_regularization=2.,early_stopping=False,random_state=random_state)
    tree=HistGradientBoostingClassifier(**params).fit(X[train],y[train],sample_weight=weights['train'])
    raw_cal=tree.predict_proba(X[cal])[:,1]
    sigmoid=SigmoidCalibration().fit(raw_cal,y[cal],weights['calibration'])
    isotonic=IsotonicRegression(out_of_bounds='clip').fit(raw_cal,y[cal],sample_weight=weights['calibration'])
    filled=np.where(np.isnan(X),mean,X); Z=(filled-mean)/scale
    logistic=LogisticRegression(C=1.,max_iter=400,random_state=random_state).fit(Z[train],y[train],sample_weight=weights['train'])
    lcal=SigmoidCalibration().fit(logistic.predict_proba(Z[cal])[:,1],y[cal],weights['calibration'])
    benign=np.flatnonzero(train&(y==0));rng=np.random.default_rng(random_state)
    if len(benign)>20000:benign=rng.choice(benign,20000,replace=False)
    anomaly=IsolationForest(n_estimators=100,max_samples=256,random_state=random_state,n_jobs=1).fit(filled[benign])
    ref=np.sort(-anomaly.score_samples(filled[benign]))
    bundle={'version':VERSION,'features':NAMES,'score_semantics':'calibrated-compromise-risk+benign-novelty-percentile',
        'risk_model':tree,'calibrator':sigmoid,'anomaly_model':anomaly,'benign_score_reference':ref,
        'mean':mean,'scale':scale,'dataset_manifest':meta,'fit_variant':'episode-host-balanced/free-agent/sigmoid',
        'runtime':{p:version(p) for p in ('numpy','scipy','scikit-learn')},'sklearn_version':version('scikit-learn')}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('preserve existing model')
    with out.open('wb') as f:pickle.dump(bundle,f)
    raw=tree.predict_proba(X[test])[:,1];p=sigmoid.predict(raw);bp=lcal.predict(logistic.predict_proba(Z[test])[:,1]);ip=isotonic.predict(raw)
    per={}
    for seed in meta['splits']['test']:
        m=episode[test]==seed
        per[str(seed)]={'hgb':metrics(y[test][m],p[m]),'logistic':metrics(y[test][m],bp[m]),'isotonic_hgb':metrics(y[test][m],ip[m])}
    report={'fit_variant':bundle['fit_variant'],'parameters':params,'splits':meta['splits'],
        'model_sha256':sha256(out),'dataset_sha256':sha256(data),'labels_sha256':sha256(labels),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_files':{p:sha256(p) for p in ('blue/training/harness_ml_redesign.py','blue/harness/calibration.py','blue/harness/features.py')},
        'per_test_episode':per,'episode_summary':episode_summary(per),
        'calibration_episode_count':len(meta['splits']['calibration']),
        'claim':'conditional reused development validation; one fit; test scores do not select calibrator or prove defensive value',
        'novelty':'legacy benign-reference percentile retained for explicit ablation; default v2 actor receives risk only'}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();fit(a.data,a.out,a.seed)
