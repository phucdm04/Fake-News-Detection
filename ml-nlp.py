# %%
import kagglehub
path = kagglehub.dataset_download("tisdang/pps-data")
print("Path to dataset files:", path)

# %%
import pickle
import os

data_path = os.path.join(path, 'ml')

# Load data
with open(os.path.join(data_path, 'train_text.pkl'), 'rb') as f:
    train_data = pickle.load(f)

with open(os.path.join(data_path, 'val_text.pkl'), 'rb') as f:
    val_data = pickle.load(f)

with open(os.path.join(data_path, 'test_text.pkl'), 'rb') as f:
    test_data = pickle.load(f)

with open(os.path.join(data_path, 'train_labels.pkl'), 'rb') as f:
    train_labels = pickle.load(f)

with open(os.path.join(data_path, 'val_labels.pkl'), 'rb') as f:
    val_labels = pickle.load(f)

with open(os.path.join(data_path, 'test_labels.pkl'), 'rb') as f:
    test_labels = pickle.load(f)

# %%
import numpy as np
train_data = np.array(train_data)
val_data = np.array(val_data)
test_data = np.array(test_data)

# %%
import random
import warnings
warnings.filterwarnings("ignore")

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_global_seed(42)

# %%
# Combine training and validation data
X_all = np.concatenate([train_data, val_data])
y_all = np.concatenate([train_labels, val_labels])

# Create test_fold: -1 = train, 0 = validation
training_fold = np.concatenate([
    -1 * np.ones(len(train_data), dtype = int),
    np.zeros(len(val_data), dtype = int)
])

# %%
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def calculate_performance(y_true, y_pred):
    print("Classification report: \n", classification_report(y_true, y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average = 'macro')
    recall = recall_score(y_true, y_pred, average = 'macro')
    f1 = f1_score(y_true, y_pred, average = 'macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# %% [markdown]
# ### Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

def log_model(X, y, training_fold = None):
    
    model = LogisticRegression(max_iter = 500)

    param_grid = [
        # L1 penalty
    {
        'penalty': ['l1'],
        'solver': ['liblinear', 'saga'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'class_weight': [None, 'balanced']
    },
        # L2 penalty
    {
        'penalty': ['l2'],
        'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'class_weight': [None, 'balanced']
    },
        # Elasticnet penalty
    {
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'class_weight': [None, 'balanced'],
        'l1_ratio': [0.05, 0.1, 0.15, 0.2, 0.25]
    }
    ]

    cv_strategy = PredefinedSplit(training_fold) if training_fold is not None else 5
    
    search = GridSearchCV(model, param_grid, cv = cv_strategy, scoring = "f1", n_jobs = -1)
    search.fit(X, y)

    print("Best parameter (CV score = %0.3f):" % search.best_score_)
    print(search.best_params_)

    return search.best_estimator_

# %%
log_reg = log_model(X_all, y_all, training_fold)

# %%
test_preds = log_reg.predict(test_data)
calculate_performance(test_labels, test_preds)

# %% [markdown]
# ### SVM

# %%
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

def svm_model(X, y, training_fold = None):
    model =  LinearSVC(max_iter = 10000)

    param_grid = [
    {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1'],
    'dual': [False],
    'loss': ['squared_hinge'],
    'class_weight': [None, 'balanced']
    }, 
    {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'dual': [False, True],
    'loss': ['hinge', 'squared_hinge'],
    'class_weight': [None, 'balanced']
    }
    ]
    
    cv_strategy = PredefinedSplit(training_fold) if training_fold is not None else 5
    
    search = GridSearchCV(model, param_grid, cv = cv_strategy, scoring = "f1", n_jobs = -1)
    search.fit(X, y)

    print("Best parameter (CV score = %0.3f):" % search.best_score_)
    print(search.best_params_)

    return search.best_estimator_

# %%
model = svm_model(X_all, y_all, training_fold)

# %%
test_preds = model.predict(test_data)
calculate_performance(test_labels, test_preds)


