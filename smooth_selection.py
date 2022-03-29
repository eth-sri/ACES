'''
- this is the core file which supports prediction and certification for an arbitrary binary  selection model for ACES
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/core.py written by Jeremy Cohen
'''

import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
from scipy.special import softmax
from scipy.stats import entropy
from torch import nn


class SmoothSelection(object):

    # initializes class
    def __init__(self, base_classifier, num_classes, sigma, boundaries):
        self.base_classifier = base_classifier # selection model (same as certification model)
        self.num_classes = num_classes # number of classes
        self.sigma = sigma # noise sigma
        self.boundaries = boundaries # boundaries to consider

    # does a certification
    def certify(self, x, n0, n, alpha, batch_size):
        self.base_classifier.eval() # make sure model is in eval mode
        
        # does selection for both selection (s) model
        counts_selections_s, _ = self._sample_noises(x, n0, batch_size)
        cAHats_s = [counts_selection.argmax().item() for counts_selection in counts_selections_s]
        
        # does estimation for both selection (s) model
        counts_estimations_s, outputs_statistics = self._sample_noises(x, n, batch_size)
        nAs_s = [counts_estimation[cAHat].item() for cAHat, counts_estimation in zip(cAHats_s, counts_estimations_s)]
        pABars_s = [self._lower_confidence_bound(nA, n, alpha) for nA in nAs_s]
        certified_radii_s = [(-1, 0.0) if pABar < 0.5
                          else (cAHat, self.sigma * norm.ppf(pABar))
                          for pABar, cAHat in zip(pABars_s, cAHats_s)]
        
        # returns answer
        return certified_radii_s, outputs_statistics

    # it is also key to implement this well (this is essentially the clean part)
    def predict(self, x, n, alpha, batch_size) -> int:
        self.base_classifier.eval()
        counts_s, _ = self._sample_noises(x, n, batch_size)
        
        # selection model (s)
        top2s_s = [count.argsort()[::-1][:2] for count in counts_s]
        count1s_s = [count[top2[0]] for count, top2 in zip(counts_s, top2s_s)]
        count2s_s = [count[top2[1]] for count, top2 in zip(counts_s, top2s_s)]
        predictions_s = [-1 if binom_test(count1, count1+count2, p=0.5) > alpha
                       else top2[0]
                       for top2, count1, count2 in zip(top2s_s, count1s_s, count2s_s)]
        
        return predictions_s

    # does the sampling
    def _sample_noises(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        with torch.no_grad():
            counts_selection = np.zeros((len(self.boundaries), 2), dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                
                outputs = self.base_classifier(batch + noise)
                m = nn.Softmax(dim=1)
                outputs = m(outputs)
                outputs = outputs.cpu().numpy()
                outputs = [output[0] for output in outputs]
                
                # computes counts for selection model
                # print('outputs: ', outputs)
                for i, boundary in enumerate(self.boundaries):
                    pred0 = sum(j > boundary for j in outputs) # 0: we do clean
                    pred1 = sum(j <= boundary for j in outputs) # 1: we do certification
                    counts_selection[i][0] = counts_selection[i][0] + pred0
                    counts_selection[i][1] = counts_selection[i][1] + pred1
                
                # computes statistics of the outputs
                outputs_statistics = [np.mean(outputs), np.std(outputs), np.min(outputs),
                                        np.percentile(outputs, 25), np.percentile(outputs, 50),
                                        np.percentile(outputs, 75), np.max(outputs)]
                    
            return counts_selection, outputs_statistics
        
    # updates counts
    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    # computes lower_confidence_bound
    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
