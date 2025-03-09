import numpy as np
import pickle as pk
import os

# ****** The test set is organized this way - correct and incorrect (0 and 1) respectively, so the predicted probabilities are also in this order ***********
ex = 1

# ***** path to the saved predicted probabilities on the test set ********
save_proba_path = r'C:\sigmoid\E{}\prob.json'.format(ex)
# save_proba_path = r'C:\\softmax\E{}\prob.json'.format(ex)

# ***** Load the saved probabilities results for all the folds *****
with open(save_proba_path, 'rb') as f:
    all_results = pk.load(f)

def separation_degree(x, y):
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")
    
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        raise ValueError("Input arrays must not be empty.")
    
    def sd(xi, yj):
        return (xi - yj) / (xi + yj) if (xi + yj) != 0 else 0

    sd_sum = sum(sd(xi, yj) for xi in x for yj in y)
    return sd_sum / (m * n)

def distance_metrics(x, y):
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")  
    numerator = np.abs(x - y)
    N = len(x)
    denominator = np.sqrt(np.sum((x - y) ** 2) / N)
    
    if denominator == 0:
        raise ValueError("Denominator evaluates to zero, resulting in undefined behavior.")
    
    D_M = numerator / denominator
    return np.mean(D_M)

SD = []
DM = []

for fold, values in all_results.items():
    print(fold)
    # print(values)

    corr_predictions = values[0::2]
    incorr_predictions = values[1::2]

    corr_pred_arr = np.array((corr_predictions), dtype=float)
    incor_pred_arr = np.array((incorr_predictions), dtype=float)

    # *** Compute the separation degree and distance metrics (sigmoid) ****
    # SD.append(round(separation_degree(incor_pred_arr, corr_pred_arr), 4))
    # DM.append(round(distance_metrics(incor_pred_arr, corr_pred_arr), 4))

    # *** Compute the separation degree and distance metrics (softmax) ****
    # *** for the SD, we switch here for softmax (corr_pred_arr first) because I choose the [:, 0] ****
    SD.append(round(separation_degree(corr_pred_arr, incor_pred_arr), 4))
    DM.append(round(distance_metrics(incor_pred_arr, corr_pred_arr), 4))

print('Separation degree for each fold {}'.format(SD))
print('Separation degree mean and std {:.4f} ({:.4f})'.format(np.mean(SD), np.std(SD)))
print('Distance metrics for each fold {}'.format(DM))
print('Distance metrics mean {:.4f} ({:.4f}) '.format(np.mean(DM), np.std(DM)))
