import numpy as np
import torch
from sklearn.metrics import (
    normalized_mutual_info_score as nmi_score,
    adjusted_rand_score as ari_score,
    adjusted_mutual_info_score as ami_score,
    homogeneity_score as hom_score,
    completeness_score as com_score,
)


class cluster_metrics:
    def __init__(self, trues, predicts):
        # Ensure numpy arrays regardless of incoming types
        self.true_label = np.asarray(trues)
        if isinstance(predicts, torch.Tensor):
            self.pred_label = predicts.detach().cpu().numpy()
        else:
            self.pred_label = np.asarray(predicts)

    def evaluateFromLabel(self, use_acc: bool = False):
        """
        Returns:
            expected_K (int): # unique classes in ground truth
            used_K (int): # unique classes in predictions
            nmi (float): Normalized Mutual Information
            hom (float): Homogeneity score
            com (float): Completeness score
            ari (float): Adjusted Rand Index
            ami (float): Adjusted Mutual Information
            pred_label (np.ndarray): raw predicted labels (unaltered)
        """
        expected_K = int(np.unique(self.true_label).size)
        used_K = int(np.unique(self.pred_label).size)

        nmi = nmi_score(self.true_label, self.pred_label)
        hom = hom_score(self.true_label, self.pred_label)
        com = com_score(self.true_label, self.pred_label)
        ari = ari_score(self.true_label, self.pred_label)
        ami = ami_score(self.true_label, self.pred_label)

        return expected_K, used_K, nmi, hom, com, ari, ami, self.pred_label