import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, auc, average_precision_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
import re
import nltk
import datetime
import pandas as pd
import requests           
import matplotlib.pyplot as plt

def aps(X, y, model):
    """
        Function to calculate PR-AUC Score based on predict_proba(X)
        where X is feature values, y is target values, and model is instantiated model variable
    """
    probs = model.predict_proba(X)[:,1]
    return average_precision_score(y, probs, pos_label='1')

def aps2(X, y, model):
    """
        Function to calculate PR-AUC Score based on decision_function(X)
        where X is feature values, y is target values, and model is instantiated model variable
    """
    probs = model.decision_function(X)
    return average_precision_score(y, probs, pos_label='1')

def auc(X, y, model):
    """
        Function to calculate ROC-AUC Score based on predict_proba(X)
        where X is feature values, y is target values, and model is instantiated model variable
    """
    probs = model.predict_proba(X)[:,1] 
    return roc_auc_score(y, probs)

def auc2(X, y, model):
    """
        Function to calculate ROC-AUC Score based on decision_function(X)
        where X is feature values, y is target values, and model is instantiated model variable
    """
    probs = model.decision_function(X)
    return roc_auc_score(y, probs)

def get_metrics_confusion(X, y, y_pred, model):
    """
        Function to get accuracy, F1, ROC-AUC, recall, precision, PR-AUC scores followed by confusion matrix
        where X is feature dataset, y is target dataset, and model is instantiated model variable
    """
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, pos_label="1")
    roc_auc = auc(X, y, model)
    rec = recall_score(y, y_pred,average="binary",pos_label="1")
    prec = precision_score(y, y_pred, pos_label='1')
    pr_auc = aps(X, y, model)

    print('Accuracy: ', acc)
    print('F1 Score: ', f1)
    print('ROC-AUC: ', roc_auc)
    print('Recall: ', rec)
    print('Precision: ', prec)
    print('PR-AUC: ', pr_auc)
    
def get_metrics_2(X, y, y_pred, model):
    """
        Function to get training and validation F1, recall, precision, PR AUC scores
        Instantiate model and pass the model into function
        Pass X_train, y_train, X_val, Y_val datasets
        Pass in calculated model.predict(X) for y_pred
    """    
    ac = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, pos_label="1")
    rc = recall_score(y, y_pred, average="binary",pos_label="1")
    pr = precision_score(y, y_pred, pos_label='1')
    rocauc = auc2(X, y, model)
    prauc = aps2(X, y, model)
    
    print('Accuracy: ', ac)
    print('F1: ', f1)
    print('Recall: ', rc)
    print('Precision: ', pr)
    print('ROC-AUC: ', rocauc)
    print('PR-AUC: ', prauc)

def ann_Evaluation(y, y_pred):
       
    ac = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    rc = recall_score(y, y_pred)
    pr = precision_score(y, y_pred)
    rocauc = roc_auc_score(y, y_pred)
    prauc = average_precision_score(y, y_pred)
    
    print('Accuracy: ', ac)
    print('F1: ', f1)
    print('Recall: ', rc)
    print('Precision: ', pr)
    print('ROC-AUC: ', rocauc)
    print('PR-AUC: ', prauc)