"""Small-parametric calibration and grouped fitting weights for ML ablations."""
import numpy as np


class SigmoidCalibration:
    """Sigmoid of the base probability logit; fitted on separate episodes."""
    def fit(self, probabilities, labels, sample_weight=None):
        from sklearn.linear_model import LogisticRegression
        p=np.clip(np.asarray(probabilities),1e-6,1-1e-6)
        self.model=LogisticRegression(C=1000.,max_iter=200).fit(
            np.log(p/(1-p)).reshape(-1,1),labels,sample_weight=sample_weight)
        return self

    def predict(self, probabilities):
        p=np.clip(np.asarray(probabilities),1e-6,1-1e-6)
        return self.model.predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]


def grouped_weights(episode, agent, host):
    """Equal total episode weight, then equal host weight within episode.

    Weighting controls fitting influence; it does NOT create independent rows.
    Only the selected training/calibration split is passed to this function.
    """
    episode=np.asarray(episode);agent=np.asarray(agent);host=np.asarray(host)
    if not len(episode) or len(episode)!=len(agent) or len(episode)!=len(host):
        raise ValueError('empty or unmatched grouping')
    keys=np.rec.fromarrays([episode,agent,host],names='episode,agent,host')
    unique,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
    episodes,epcounts=np.unique(unique['episode'],return_counts=True)
    sizes=dict(zip(episodes,epcounts))
    w=1./counts[inverse]/np.asarray([sizes[s] for s in episode])
    return w/w.mean()
