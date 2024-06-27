import numpy as np
from sklearn.metrics import roc_curve

def calculate_apcer(scores, labels, threshold):
    apcer = np.mean((scores >= threshold) & (labels == 0))
    return apcer

def calculate_bpcer(scores, labels, threshold):
    bpcer = np.mean((scores < threshold) & (labels == 1))
    return bpcer

def calculate_acer(apcer, bpcer):
    return (apcer + bpcer) / 2

def calculate_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    return eer

def calculate_tpr_at_fpr(scores, labels, target_fpr=1e-4):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    tpr_at_target_fpr = np.max(tpr[fpr <= target_fpr])
    return tpr_at_target_fpr

def calculate_hter(apcer, bpcer):
    return (apcer + bpcer) / 2
