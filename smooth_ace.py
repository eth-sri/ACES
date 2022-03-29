'''
- this is the core file which supports prediction and certification for ACES
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/core.py written by Jeremy Cohen
'''

import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
from scipy.special import softmax
from scipy.stats import entropy


class SmoothACE(object):
    
    ABSTAIN = -1

    def __init__(self, base_classifier, num_classes, sigma, boundaries):
        self.base_classifier = base_classifier # certification model and also leads to the entropy-based selection model
        self.num_classes = num_classes # number of classes
        self.sigma = sigma # noise term
        self.boundaries = boundaries # entropy boundaries (thresholds) to consider

    def certify(self, x, n0, n, alpha, batch_size):
        self.base_classifier.eval()
        
        # does selection for both selection (s) and certification (c) model
        counts_selections_s, counts_selections_c, _ = self._sample_noises(x, n0, batch_size)
        cAHats_s = [counts_selection.argmax().item() for counts_selection in counts_selections_s]
        cAHats_c = [counts_selection.argmax().item() for counts_selection in counts_selections_c]
        
        counts_estimations_s, counts_estimations_c, entropies_statistics = self._sample_noises(x, n, batch_size)
        nAs_s = [counts_estimation[cAHat].item() for cAHat, counts_estimation in zip(cAHats_s, counts_estimations_s)]
        pABars_s = [self._lower_confidence_bound(nA, n, alpha) for nA in nAs_s]
        certified_radii_s = [(SmoothACE.ABSTAIN, 0.0) if pABar < 0.5
                          else (cAHat, self.sigma * norm.ppf(pABar))
                          for pABar, cAHat in zip(pABars_s, cAHats_s)]
        nAs_c = [counts_estimation[cAHat].item() for cAHat, counts_estimation in zip(cAHats_c, counts_estimations_c)]
        pABars_c = [self._lower_confidence_bound(nA, n, alpha) for nA in nAs_c]
        certified_radii_c = [(SmoothACE.ABSTAIN, 0.0) if pABar < 0.5
                          else (cAHat, self.sigma * norm.ppf(pABar))
                          for pABar, cAHat in zip(pABars_c, cAHats_c)]
        
        return certified_radii_s, certified_radii_c, entropies_statistics

    def predict(self, x, n, alpha, batch_size) -> int:
        self.base_classifier.eval()
        counts_s, counts_c, _ = self._sample_noises(x, n, batch_size)
        
        # selection model (s)
        top2s_s = [count.argsort()[::-1][:2] for count in counts_s]
        count1s_s = [count[top2[0]] for count, top2 in zip(counts_s, top2s_s)]
        count2s_s = [count[top2[1]] for count, top2 in zip(counts_s, top2s_s)]
        predictions_s = [SmoothACE.ABSTAIN if binom_test(count1, count1+count2, p=0.5) > alpha
                       else top2[0]
                       for top2, count1, count2 in zip(top2s_s, count1s_s, count2s_s)]
        
        # certification model (c)
        top2s_c = [count.argsort()[::-1][:2] for count in counts_c]
        count1s_c = [count[top2[0]] for count, top2 in zip(counts_c, top2s_c)]
        count2s_c = [count[top2[1]] for count, top2 in zip(counts_c, top2s_c)]
        predictions_c = [SmoothACE.ABSTAIN if binom_test(count1, count1+count2, p=0.5) > alpha
                       else top2[0]
                       for top2, count1, count2 in zip(top2s_c, count1s_c, count2s_c)]
        
        return predictions_s, predictions_c

    def _sample_noises(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        with torch.no_grad():
            counts_selection = np.zeros((len(self.boundaries), 2), dtype=int)
            counts_certification = np.zeros((len(self.boundaries), self.num_classes), dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                
                outputs = self.base_classifier(batch + noise)
                outputs = outputs.cpu().numpy()   
                
                # computes counts for entropy-based selection model
                entropies = [entropy(softmax(j), base=self.num_classes) for j in outputs]
                for i, boundary in enumerate(self.boundaries):
                    pred0 = sum(j > boundary for j in entropies) # 0: core model chosen
                    pred1 = sum(j <= boundary for j in entropies) # 1: certification model chosen
                    counts_selection[i][0] = counts_selection[i][0] + pred0
                    counts_selection[i][1] = counts_selection[i][1] + pred1
                    
                # computes counts for certification model
                predictions = outputs.argmax(1)
                counts_certification += self._count_arr(predictions, self.num_classes)
                
                # computes entropy statistics
                entropies_statistics = [np.mean(entropies), np.std(entropies), np.min(entropies),
                                        np.percentile(entropies, 25), np.percentile(entropies, 50),
                                        np.percentile(entropies, 75), np.max(entropies)]
                    
            return counts_selection, counts_certification, entropies_statistics

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    