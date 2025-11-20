"""
Class Imbalance Handling Solutions for Network Intrusion Detection

This module provides various techniques to handle class imbalance in the dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter


def handle_class_imbalance(X_train, y_train, method='smote', random_state=42):
    """
    Apply class imbalance handling technique.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    method : str
        Method to use: 'smote', 'adasyn', 'borderline_smote', 
                      'smote_tomek', 'smote_enn', 'undersample', 'class_weight'
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    X_resampled, y_resampled : resampled training data
    """
    
    print(f"Original class distribution: {Counter(y_train)}")
    
    if method == 'smote':
        sampler = SMOTE(random_state=random_state, k_neighbors=3)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=random_state, n_neighbors=3)
    elif method == 'borderline_smote':
        sampler = BorderlineSMOTE(random_state=random_state, k_neighbors=3)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=random_state)
    elif method == 'smote_enn':
        sampler = SMOTEENN(random_state=random_state)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    print(f"Resampled class distribution: {Counter(y_resampled)}")
    
    return X_resampled, y_resampled


def compare_imbalance_methods(X_train, y_train, X_test, y_test, methods=None):
    """
    Compare different class imbalance handling methods.
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : test data
    methods : list of methods to compare
    
    Returns:
    --------
    results_df : DataFrame with comparison results
    """
    
    if methods is None:
        methods = ['none', 'smote', 'adasyn', 'borderline_smote', 'class_weight']
    
    results = []
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing method: {method.upper()}")
        print(f"{'='*50}")
        
        if method == 'none':
            X_train_balanced = X_train
            y_train_balanced = y_train
            class_weight = None
        elif method == 'class_weight':
            X_train_balanced = X_train
            y_train_balanced = y_train
            class_weight = 'balanced'
        else:
            X_train_balanced, y_train_balanced = handle_class_imbalance(
                X_train, y_train, method=method
            )
            class_weight = None
        
        # Train model
        model = DecisionTreeClassifier(
            random_state=42,
            class_weight=class_weight
        )
        model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        results.append({
            'Method': method.upper(),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision (weighted)': precision_score(y_test, y_pred, average='weighted'),
            'Recall (weighted)': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score (weighted)': f1_score(y_test, y_pred, average='weighted'),
            'F1-Score (macro)': f1_score(y_test, y_pred, average='macro'),
            'Training Samples': len(X_train_balanced)
        })
        
        # Print per-class metrics
        print("\nPer-class F1-scores:")
        report = classification_report(y_test, y_pred, output_dict=True)
        for i, class_name in enumerate(['BotAttack', 'Normal', 'PortScan']):
            if str(i) in report:
                print(f"  {class_name}: {report[str(i)]['f1-score']:.4f}")
    
    results_df = pd.DataFrame(results)
    return results_df


def cross_validate_with_imbalance_handling(X, y, method='smote', cv=5):
    """
    Perform cross-validation with class imbalance handling.
    
    Parameters:
    -----------
    X, y : full dataset
    method : imbalance handling method
    cv : number of folds
    
    Returns:
    --------
    cv_scores : cross-validation scores
    """
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{cv}")
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Handle imbalance
        if method != 'none':
            X_train_fold, y_train_fold = handle_class_imbalance(
                X_train_fold, y_train_fold, method=method
            )
        
        # Train and evaluate
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_val_fold)
        
        score = f1_score(y_val_fold, y_pred_fold, average='weighted')
        scores.append(score)
        print(f"  F1-Score: {score:.4f}")
    
    print(f"\n{'='*50}")
    print(f"Cross-Validation Results ({method.upper()})")
    print(f"{'='*50}")
    print(f"Mean F1-Score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    return scores


# Example usage:
if __name__ == "__main__":
    # This is a template - integrate with your actual data loading
    print("""
    Example Usage:
    
    # 1. Load your data
    df = pd.read_csv("Network_logs.csv")
    # ... preprocessing ...
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 3. Compare different methods
    results = compare_imbalance_methods(X_train, y_train, X_test, y_test)
    print(results)
    
    # 4. Use best method
    X_train_balanced, y_train_balanced = handle_class_imbalance(
        X_train, y_train, method='smote'
    )
    
    # 5. Train final model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_balanced, y_train_balanced)
    """)

