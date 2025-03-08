import numpy as np
import pickle as pk
import os

# Note: 3 things: 
# (a) folder_path (the path to the exercise folder. e.g., E1, E2, etc), 
# (b) save_proba_path (the path to the saved probabilites for the corresponding exercise, e.g., E1, E2.). Exercise must match as in (i) for folder path 
# (c) 'Values' list in the code (line 29 and 30), depending on either softmax or sigmoid. 

ex =1
folder_path = r'C:\EHE_Dataset\data_and_features\Extracted_Features\cv_cs\xyz\{}'.format(ex)

# ***** Sigmoid *****
# save_proba_path = r'\sigmoid\E{}\prob.json'.format(ex)

# ***** Softmax *****
save_proba_path = r'\softmax\E{}\prob.json'.format(ex)

# ***** Load the saved probabilities results for all the folds *****
with open(save_proba_path, 'rb') as f:
    all_results = pk.load(f)

# ***** the raw human score from the paper (10-x) to match the labeling of AD as 0 and non-AD as 1****
values = [0, 7, 8, 0, 5, 10, 0, 0, 0, 10, 5, 6, 4, 8, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# **** Note: 10-val for sigmoid and val for softmax *****
# values = [10-val for val in values]
values = [val for val in values]

min_val, max_val = min(values), max(values)
normalized_values = [(val - min_val) / (max_val - min_val) if max_val > min_val else 0 for val in values]

def cross_correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    
    if denominator == 0:
        return 0    
    return np.abs(numerator / denominator)

ED = []
CR = []
for fold in range(len(os.listdir(folder_path))):

    y_test = np.load(os.path.join(folder_path, str(fold+1), 'eval_label.pkl'), allow_pickle=True)

    # normalized human evaluation score evaluation for this fold 
    human_eval_score_norm = [normalized_values[(int(subj.split('S')[1][:2]))-1] for subj in y_test[0]]

    # **** Get the predicted probabilities for the corresponding fold and to 4dp. ****
    predicted_prob = np.vectorize(lambda x: f"{x:.4f}")(all_results[fold+1])

    eval_score_arr = np.array((human_eval_score_norm), dtype=float)
    predicted_prob_arr = np.array((predicted_prob), dtype=float)

    # *** Compute the euclidean distance and cross correlation ****
    ED.append(round(np.sqrt(np.sum((eval_score_arr - predicted_prob_arr)**2)), 4))
    CR.append(round(cross_correlation(eval_score_arr, predicted_prob_arr), 4))

    # print(np.linalg.norm(eval_score_arr - predicted_prob_arr))

print('Eucliean distance for each fold {}'.format(ED))
print('Euclidean distance mean {:.4f}'.format(np.mean(ED)))
print('Cross correlation for each fold {}'.format(CR))
print('Cross correlation mean {:.4f}'.format(np.mean(CR)))

