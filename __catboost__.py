#  ____  _   _    _    ____  _____     _ 
# |  _ \| | | |  / \  / ___|| ____|   / |
# | |_) | |_| | / _ \ \___ \|  _|     | |
# |  __/|  _  |/ ___ \ ___) | |___    | |
# |_|   |_| |_/_/   \_\____/|_____|   |_| 

# Importing of Libraries into the File
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

# Machine Learning algorithm imports
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve

# CatBoost algorithm
from catboost import CatBoostClassifier

# Importing the data set
df=pd.read_excel("/Users/_.rohan._/Desktop/EMPLOYEE_ATTRITION/EXTRA datasets/EA with names.xlsx")
df = df.iloc[:, 1:]

# Total Count of Employee
def Attrition_Total():
    print("Total employees:",df.shape[0])

Attrition_Total()

#Attrition_Percentage()

# Attrition YES|NO Count
def Attrition_Count():
    attrition_counts = df['Attrition'].value_counts()
    print(attrition_counts)

Attrition_Count()

# Noise values elimination
dummy_col = [column for column in df.drop('Attrition', axis=1).columns if df[column].nunique() < 20]
data = pd.get_dummies(df, columns=dummy_col, drop_first=True, dtype='uint8')
df['Attrition'] = df['Attrition'].map({'No': 0, 'Yes': 1})
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# CatBoost application
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

cb_clf = CatBoostClassifier()
cb_clf.fit(X_train, y_train)

# evaluate function for ML algorithms
def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    print("TRAINING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

    print("TESTING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    
evaluate(cb_clf, X_train, X_test, y_train, y_test)

# Data Points in Order column in operation
stay = y_train.value_counts()[0] / y_train.shape[0]
leave = y_train.value_counts()[1] / y_train.shape[0]

#Graph comparison of precision and threshold
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.title("Precision/Recall Tradeoff")

# Data classification for feature importance
def feature_imp(df, model):
    fi = pd.DataFrame()
    fi["feature"] = df.columns
    fi["importance"] = model.feature_importances_
    return fi.sort_values(by="importance", ascending=False)

scores_dict= {
        'Train': roc_auc_score(y_train, cb_clf.predict(X_train)),
        'Test': roc_auc_score(y_test, cb_clf.predict(X_test)),
    }

# ROC curve function
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    
# CatBoost algorithm for data points in order column with ROC accuracy score of 0.620

cb_clf = CatBoostClassifier()
cb_clf.fit(X_train, y_train)
evaluate(cb_clf, X_train, X_test, y_train, y_test)

# Precision, recall and threshold curve graphs
def graphs():
    precisions, recalls, thresholds = precision_recall_curve(y_test, cb_clf.predict(X_test))
    plt.figure(figsize=(14, 25))
    plt.subplot(4, 2, 1)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    plt.subplot(4, 2, 2)
    plt.plot(precisions, recalls)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("PR Curve: precisions/recalls tradeoff")

    plt.subplot(4, 2, 3)
    fpr, tpr, thresholds = roc_curve(y_test, cb_clf.predict(X_test))
    plot_roc_curve(fpr, tpr)

df = feature_imp(X, cb_clf)[:35]
df.set_index('feature', inplace=True)
df.plot(kind='barh', figsize=(8, 10))
plt.title('Feature Importance')

# display graphs
plt.show()

# Extracting Y-axis coordinate points
print(feature_imp(X,cb_clf))

# Attrition YES|NO Count
# Attrition Rate from the Training and Testing Dataset
def Attrition_Percentage(): 
    stay = (y_train.value_counts()[0] / y_train.shape) [0]
    leave = (y_train.value_counts()[1] / y_train.shape)[0]
    print("===============TRAIN=================")
    print(f"Staying Rate: {stay * 100:.2f}%")
    print(f"Leaving Rate: {leave * 100 : .2f}%")
    
    stay = (y_test.value_counts()[0] / y_test.shape)[0]
    leave = (y_test.value_counts()[1] / y_test. shape) [0]
    print("===============TEST=================")
    print(f"Staying Rate: {stay * 100:.2f}%")
    print(f"Leaving Rate: {leave * 100 : .2f}%")

#  ____  _   _    _    ____  _____       ____  
# |  _ \| | | |  / \  / ___|| ____|     |___ \ 
# | |_) | |_| | / _ \ \___ \|  _|         __) |
# |  __/|  _  |/ ___ \ ___) | |___       / __/ 
# |_|   |_| |_/_/   \_\____/|_____|     |_____|

# Attrition based on certain employee

# __PHASE2__ -------------->>>>>>>>

import seaborn as sns

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Calling the function for training and testing data
y_train_pred = cb_clf.predict(X_train)
y_test_pred = cb_clf.predict(X_test)

# Plot confusion matrix for training data
plot_confusion_matrix(y_train, y_train_pred)

# Plot confusion matrix for testing data
plot_confusion_matrix(y_test, y_test_pred)
