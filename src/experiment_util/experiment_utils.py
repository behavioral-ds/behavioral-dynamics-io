import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

import sys

def reshape_to_n_50_2(arr):
    # Do nothing if there is < 50 activities
    if arr.shape[0] < 50:
        return arr[np.newaxis, :, :]  # Shape (1, k, 2)
    total_rows = arr.shape[0]
    # Drop rows that don't fit into 50-row blocks
    usable_rows = (total_rows // 50) * 50
    trimmed = arr[:usable_rows]
    reshaped = trimmed.reshape(-1, 50, 2)
    return reshaped

def run_experiment(df_pos, df_neg, seed):

    df_sample = pd.concat([df_pos, df_neg])

    df_sample['flattened'] = df_sample['feature_col'].apply(lambda x: x.flatten())
    X = np.stack(df_sample['flattened'].values)  # Convert to 2D array
    y = df_sample['russian']
    users = df_sample['user'].values  # Track users

    # Set up KFold cross-validation
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    all_y_true = []
    all_rf_preds = []
    all_xgb_preds = []

    misclassified_users_rf = defaultdict(int)
    misclassified_users_xgb = defaultdict(int)


    for train_index, test_index in kf.split(X, y):
        # Split data into train and test sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        test_users = users[test_index]  # Get the test users

        all_y_true.extend(y_test)
        
        # Ensure training data has both classes
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            print(f"Skipping fold: Only one class present in y_train ({unique_classes})")
            continue  # Skip this fold if it contains only one class

        # Random Forest Classifier
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)

        all_rf_preds.extend(rf_predictions)

        for idx, user in enumerate(test_users):
            if rf_predictions[idx] != y_test.iloc[idx]:  # If prediction is wrong
                misclassified_users_rf[user] += 1

        # XGBoost Classifier
        xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        xgb_predictions = xgb_model.predict(X_test)

        all_xgb_preds.extend(xgb_predictions)

        for idx, user in enumerate(test_users):
            if xgb_predictions[idx] != y_test.iloc[idx]:  # If prediction is wrong
                misclassified_users_xgb[user] += 1

    # Ensure confusion matrices are only computed when predictions exist
    if all_rf_preds and all_xgb_preds:

        confusion_matrix_rf = confusion_matrix(all_y_true, all_rf_preds)
        confusion_matrix_xgb = confusion_matrix(all_y_true, all_xgb_preds)
        overall_rf_f1 = f1_score(all_y_true, all_rf_preds, average='macro')
        overall_xgb_f1 = f1_score(all_y_true, all_xgb_preds, average='macro')


        print(f"Confusion Matrix (Random Forest):")
        print(confusion_matrix_rf)
        print(f"Confusion Matrix (XGBoost):")
        print(confusion_matrix_xgb)
        print(f"F1 Score (Random Forest): {overall_rf_f1:.2f}")
        print(f"F1 Score (XGBoost): {overall_xgb_f1:.2f}")
        print(f"\n")

        return confusion_matrix_rf, confusion_matrix_xgb, overall_rf_f1, overall_xgb_f1, misclassified_users_rf, misclassified_users_xgb
        
    
    else:
        print(f"Run {i}: Skipped due to class imbalance (No valid folds).")
    
    
   


def run_experiment_n_runs(df_pos, df_neg, sampling_range,n=25):
    running_confusion_matrix_rf = np.zeros((2, 2), dtype=int)
    running_confusion_matrix_xgb = np.zeros((2, 2), dtype=int)
    
    misclassified_users_rf_all_subs = defaultdict(int)
    misclassified_users_xgb_all_subs = defaultdict(int)
    
    for i in range(1, n+1):
        seed = 100 + i
        # random_users = df.sample(n=64, random_state=seed).reset_index(drop=True)
    
        df_russian_sampled, df_sampled = matched_window(df_pos, df_neg) #matched_sampling_with_tolerance(df_pos, df_neg,r=sampling_range)
        
        df_sample = pd.concat([df_russian_sampled, df_sampled])
    
        df_sample['flattened'] = df_sample['feature_col'].apply(lambda x: x.flatten())
        X = np.stack(df_sample['flattened'].values)  # Convert to 2D array
        y = df_sample['russian']
        users = df_sample['user'].values  # Track users
    
        # Set up KFold cross-validation
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    
        all_y_true = []
        all_rf_preds = []
        all_xgb_preds = []
    
        for train_index, test_index in kf.split(X, y):
            # Split data into train and test sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            test_users = users[test_index]  # Get the test users
    
            all_y_true.extend(y_test)
            
            # Ensure training data has both classes
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                print(f"Skipping fold: Only one class present in y_train ({unique_classes})")
                continue  # Skip this fold if it contains only one class
    
            # Random Forest Classifier
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X_train, y_train)
            rf_predictions = rf_model.predict(X_test)
    
            all_rf_preds.extend(rf_predictions)
    
            for idx, user in enumerate(test_users):
                if rf_predictions[idx] != y_test.iloc[idx]:  # If prediction is wrong
                    misclassified_users_rf_all_subs[user] += 1
    
            # XGBoost Classifier
            xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            xgb_model.fit(X_train, y_train)
            xgb_predictions = xgb_model.predict(X_test)
    
            all_xgb_preds.extend(xgb_predictions)
    
            for idx, user in enumerate(test_users):
                if xgb_predictions[idx] != y_test.iloc[idx]:  # If prediction is wrong
                    misclassified_users_xgb_all_subs[user] += 1
    
        # Ensure confusion matrices are only computed when predictions exist
        if all_rf_preds and all_xgb_preds:
            print(f"Run {i}: Confusion Matrix (Random Forest):")
            print(confusion_matrix(all_y_true, all_rf_preds))
    
            print(f"Run {i}: Confusion Matrix (XGBoost):")
            print(confusion_matrix(all_y_true, all_xgb_preds))
            overall_rf_f1 = f1_score(all_y_true, all_rf_preds, average='macro')
            overall_xgb_f1 = f1_score(all_y_true, all_xgb_preds, average='macro')
            print(f"Run {i}: Overall F1 Score (Random Forest): {overall_rf_f1:.2f}")
            print(f"Run {i}: Overall F1 Score (XGBoost): {overall_xgb_f1:.2f}")
    
            print(f"\n")
            running_confusion_matrix_rf += confusion_matrix(all_y_true, all_rf_preds)
            running_confusion_matrix_xgb += confusion_matrix(all_y_true, all_xgb_preds)
        
        else:
            print(f"Run {i}: Skipped due to class imbalance (No valid folds).")
    
    
    print("Final Confusion Matrix (Random Forest):\n", running_confusion_matrix_rf)    
    
    # Extract values
    TN, FP, FN, TP = running_confusion_matrix_rf.ravel()
    # Reconstruct y_true and y_pred (each element occurs its respective count times)
    y_true = np.array([0] * (TN + FP) + [1] * (FN + TP))  # 0 for negative class, 1 for positive class
    y_pred = np.array([0] * TN + [1] * FP + [0] * FN + [1] * TP)  # Predictions
    
    # Compute F1-score
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score (Random Forest): {f1:.2f}\n")
    
    print("Final Confusion Matrix (XGBoost):\n", running_confusion_matrix_xgb)        
    
    # Extract values
    TN, FP, FN, TP = running_confusion_matrix_xgb.ravel()
    # Reconstruct y_true and y_pred (each element occurs its respective count times)
    y_true = np.array([0] * (TN + FP) + [1] * (FN + TP))  # 0 for negative class, 1 for positive class
    y_pred = np.array([0] * TN + [1] * FP + [0] * FN + [1] * TP)  # Predictions
    
    # Compute F1-score
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score (XGBoost): {f1:.2f}\n")

def remove_agreement_action(traj):
    # map reply action to single action without agreement
    traj = traj.copy().reshape(-1, 2)
    traj[traj[:, 1] == 5, 1] = 3
    traj[traj[:, 1] == 4, 1] = 3
    return reshape_to_n_50_2(traj)

def remove_agreement_states(traj):
    # map reply action to single action without agreement
    traj = traj.copy().reshape(-1, 2)
    # get reply states
    traj[traj[:, 0] == 1, 0] = 0
    traj[traj[:, 0] == 2, 0] = 0
    # initial reply states
    traj[traj[:, 0] == 6, 0] = 5
    traj[traj[:, 0] == 7, 0] = 5
    # engaged reply states
    traj[traj[:, 0] == 10, 0] = 9
    traj[traj[:, 0] == 11, 0] = 9

    # decrement to get rid of gaps in the states

    # get reply states
    traj[traj[:, 0] > 5, 0] -= 2
    traj[traj[:, 0] > 0, 0] -= 2

    return reshape_to_n_50_2(traj)

def remove_agreement(traj):
    return remove_agreement_states(remove_agreement_action(traj))
