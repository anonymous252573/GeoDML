import numpy as np
import pickle as pk
import os
import pandas as pd
import warnings
from itertools import islice
import math

# ************ (1) Obtain the Normalized clinical evaluation score **************
# Suppress openpyxl warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Initialize dictionaries
TS_clinical_score_dict = {}
PO_clinical_score_dict = {}
CF_clinical_score_dict = {}

# Mapping for folder prefixes
folder_prefix_map = {
    "Expert": "G001",
    "NotExpert": "G002",
    "BackPain": "G003",
    "Parkinson": "G004",
    "Stroke": "G005",
}

# Root directory of the folder structure
root_dir = r"C:\KIMORE_Dataset\Kimore"

# Function to format subject name
def format_subject_name(prefix, subject_id):
    if subject_id > 9:
        return f"{prefix}S{subject_id:03d}"
    else:
        return f"{prefix}S{subject_id:03d}"

# Loop through the folders
for main_folder, prefix in folder_prefix_map.items():
    base_path = os.path.join(root_dir, "GPP" if main_folder in ["Stroke", "Parkinson", "BackPain"] else "CG", main_folder)

    if not os.path.exists(base_path):
        continue

    for subject_folder in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject_folder)

        # Check if it's a valid directory
        if not os.path.isdir(subject_path):
            continue

        # Extract subject ID
        subject_id = int(subject_folder.split("_")[-1][2:])  # e.g., S_ID1 -> 1 or P_ID1 -> 1

        # Format subject name
        subject_name = format_subject_name(prefix, subject_id)

        # Go to folder
        es1_path = os.path.join(subject_path, "ES1")
        if not os.path.exists(es1_path):
            continue

        # Go to Label folder
        label_path = os.path.join(es1_path, "Label")
        if not os.path.exists(label_path):
            continue

        # Locate the specific Excel file
        expected_filename = f"ClinicalAssessment_{subject_folder}.xlsx"
        excel_path = os.path.join(label_path, expected_filename)

        if os.path.exists(excel_path):
            # Read the Excel file
            data = pd.read_excel(excel_path)

            # Extract column values
            TS_values = data.iloc[0, 1:6].tolist()  # Columns 2-6 (1-based index)
            PO_values = data.iloc[0, 6:11].tolist()  # Columns 7-11
            CF_values = data.iloc[0, 11:16].tolist()  # Columns 12-16

            # Populate dictionaries
            TS_clinical_score_dict[subject_name] = TS_values
            PO_clinical_score_dict[subject_name] = PO_values
            CF_clinical_score_dict[subject_name] = CF_values

# Normalization function
def normalize_dicts(dict1, dict2, dict3):
    all_values = [val for d in (dict1, dict2, dict3) for lst in d.values() for val in lst]
    min_val, max_val = min(all_values), max(all_values)
    
    if max_val == min_val:
        return dict1, dict2, dict3
    
    def normalize_value(lst):
        return [(val - min_val) / (max_val - min_val) for val in lst]
    
    norm_dict1 = {k: normalize_value(v) for k, v in dict1.items()}
    norm_dict2 = {k: normalize_value(v) for k, v in dict2.items()}
    norm_dict3 = {k: normalize_value(v) for k, v in dict3.items()}
    
    return norm_dict1, norm_dict2, norm_dict3

TS_clinical_score_dict_normalized, PO_clinical_score_dict_normalized, CF_clinical_score_dict_normalized \
    = normalize_dicts(TS_clinical_score_dict, PO_clinical_score_dict, CF_clinical_score_dict)

# *************** End here ***************** 

# ************ (2) Then perform the cross correlation computation for each fold and exercise **************

mapp = {0:[2], 1:[3,4], 2:[5,6], 3:[7,8], 4:[9]} # as stored in folder, index 0 for E2, 1 for E3 and 4, etc
ex1 = next(iter(mapp))
ex2 = next(islice(mapp.keys(), 1, None))
ex3 = next(islice(mapp.keys(), 2, None))
ex4 = next(islice(mapp.keys(), 3, None))
ex5 = next(islice(mapp.keys(), 4, None))

# ** YOU ONLY CHANGE FOR THE EXERCISE HERE (ex1, ex2, etc) AND THE index [0] or [1] FOR EX2, EX3 and EX4 BECAUSE THEY HAVE R & L Segmented Version 
#  and comment the part needed for Softmax in the code below ****

ind_ = ex5   # change for ex1, ex2, etc
folder_sel = mapp[ind_][0]  # for ex2, ex3, and ex4, do for both [0] and [1]

print("COMPUTING RESULTS FOR folder E{} belonging to Exercise {}".format(folder_sel, ind_+1))

# ***** Softmax *****
folder_path = r'C:\KIMORE_Dataset\data_and_features\Extracted_Features\cv_cs\xyz\{}'.format(folder_sel)
save_proba_path = r'C:\softmax\E{}\prob.json'.format(folder_sel)  #path of the saved probabiites of the test set

# ***** Sigmoid *****
# folder_path = r'C:\KIMORE_Dataset\data_and_features\Extracted_Features\cv_cs\xyz\{}'.format(folder_sel)
# save_proba_path = r'C:\sigmoid\E{}\prob.json'.format(folder_sel)  #path of the saved probabiites of the test set

# ***** Load the saved probabilities results for all the folds for the chosen exercise *****
with open(save_proba_path, 'rb') as f:
    all_results = pk.load(f)

def cross_correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    
    if denominator == 0:
        return 0    
    return np.abs(numerator / denominator)

CR_TS = []
CR_PO = []
CR_CF = []

for fold in range(len(os.listdir(folder_path))):
    y_test = np.load(os.path.join(folder_path, str(fold+1), 'eval_label.pkl'), allow_pickle=True)
    y_test = y_test[0]

    # normalized human evaluation score evaluation for this fold. For some subjects with no clinical score, we ignore and no computation
    TS_normalized_score = []
    TS_nan_indices = []
    for idx, item in enumerate(y_test):
        key = item[:8] 
        value = TS_clinical_score_dict_normalized.get(key, [None])[ind_]    
        if value is not None and not math.isnan(value): 
            TS_normalized_score.append(value)
        else:
            TS_nan_indices.append(idx) 

    PO_normalized_score = []
    PO_nan_indices = []
    for idx, item in enumerate(y_test):
        key = item[:8] 
        value = PO_clinical_score_dict_normalized.get(key, [None])[ind_]    
        if value is not None and not math.isnan(value): 
            PO_normalized_score.append(value)
        else:
            PO_nan_indices.append(idx) 
            
    CF_normalized_score = []
    CF_nan_indices = []
    for idx, item in enumerate(y_test):
        key = item[:8] 
        value = CF_clinical_score_dict_normalized.get(key, [None])[ind_]    
        if value is not None and not math.isnan(value): 
            CF_normalized_score.append(value)
        else:
            CF_nan_indices.append(idx) 

    # ****** UNCOMMENT THIS PART FOR SOFTMAX (VERY IMPORTANT!!!!!) *******
    TS_normalized_score = [1-val for val in TS_normalized_score]
    PO_normalized_score = [1-val for val in PO_normalized_score]
    CF_normalized_score = [1-val for val in CF_normalized_score]

    # **** Get the predicted probabilities for the corresponding fold and to 4dp. ****
    predicted_prob = np.vectorize(lambda x: f"{x:.4f}")(all_results[fold+1])

    # **** Remove the probab values that does have the eval score ****
    TS_pred_prob = [item for idx, item in enumerate(predicted_prob) if idx not in TS_nan_indices]
    PO_pred_prob = [item for idx, item in enumerate(predicted_prob) if idx not in PO_nan_indices]
    CF_pred_prob = [item for idx, item in enumerate(predicted_prob) if idx not in CF_nan_indices]

    eval_score_ts = np.array((TS_normalized_score), dtype=float)
    eval_score_po = np.array((PO_normalized_score), dtype=float)
    eval_score_cf = np.array((CF_normalized_score), dtype=float)

    predicted_prob_arr_ts = np.array((TS_pred_prob), dtype=float)
    predicted_prob_arr_po = np.array((PO_pred_prob), dtype=float)
    predicted_prob_arr_cf = np.array((CF_pred_prob), dtype=float)

    CR_TS.append(round(cross_correlation(eval_score_ts, predicted_prob_arr_ts), 4))
    CR_PO.append(round(cross_correlation(eval_score_po, predicted_prob_arr_po), 4))
    CR_CF.append(round(cross_correlation(eval_score_cf, predicted_prob_arr_cf), 4))

print('--------------------------------------------------------')
print('Cross correlation TS for each fold {}'.format(CR_TS))
print('Cross correlation TS mean {:.4f}'.format(np.mean(CR_TS)))
print('--------------------------------------------------------')

print('--------------------------------------------------------')
print('Cross correlation PO for each fold {}'.format(CR_PO))
print('Cross correlation PO mean {:.4f}'.format(np.mean(CR_PO)))
print('--------------------------------------------------------')

print('--------------------------------------------------------')
print('Cross correlation CF for each fold {}'.format(CR_CF))
print('Cross correlation CF mean {:.4f}'.format(np.mean(CR_CF)))
print('--------------------------------------------------------')