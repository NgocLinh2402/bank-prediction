#%%
# importing required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

url = 'https://raw.githubusercontent.com/ashutoshmakone/Bank-Marketing-Dataset-Machine-Learning/main/bank.csv'
df = pd.read_csv(url)

#Encode deposit to binary
df['deposit'].replace(to_replace='yes', value=1, inplace=True)
df['deposit'].replace(to_replace='no',  value=0, inplace=True)
df['deposit'].head()

#Encode loan to binary
df['loan'].replace(to_replace='no', value=1, inplace=True)
df['loan'].replace(to_replace='yes',  value=0, inplace=True)
df.head()

#encode default
df['default'].replace(to_replace='no', value=1, inplace=True)
df['default'].replace(to_replace='yes',  value=0, inplace=True)
df.head()

#Encode housing
df['housing'].replace(to_replace='no', value=1, inplace=True)
df['housing'].replace(to_replace='yes',  value=0, inplace=True)
df.head()

dummies = pd.get_dummies(df['marital']).astype(int)
df = df.drop(['marital'], axis = 1)
df = df.join(dummies)
df.head()

#Encode education

dummies = pd.get_dummies(df['education']).astype(int)
df = df.drop('education',axis = 1)
df = df.join(dummies)
# Check if 'unknown' column exists before dropping
if 'unknown' in df.columns:
    df = df.drop('unknown',axis = 1)
df.head()

#Encode job
dummies = pd.get_dummies(df['job']).astype(int)
df = df.drop('job',axis = 1)
df = df.join(dummies)
if 'unknown' in df.columns:
    df = df.drop('unknown',axis = 1)
df.head()

#Encode contact
dummies = pd.get_dummies(df['contact']).astype(int)
df = df.drop('contact',axis = 1)
df = df.join(dummies)
if 'unknown' in df.columns:
    df = df.drop('unknown',axis = 1)
df.head()

#Encode month
dummies = pd.get_dummies(df['month']).astype(int)
df = df.drop('month',axis = 1)
df = df.join(dummies)
if 'dec' in df.columns:
    df = df.drop('dec',axis = 1)
df.head()

#Encode poutcome
dummies = pd.get_dummies(df['poutcome']).astype(int)
df = df.drop('poutcome',axis = 1)
df = df.join(dummies)
if 'other' in df.columns:
    df = df.drop('other',axis = 1)
df.head()

#%%

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

# Modeling
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

import gdown

# Link file Google Drive
file_id = '1sAO6Dq7Lpj78qe8w_OmZNjs9h9L97mhs'
url = f'https://drive.google.com/uc?id={file_id}'

# Tải file về
gdown.download(url, 'PreprocessedBank.csv', quiet=False)

#Reading Preprocessed data. First column is deleted because its index and redundant
df = pd.read_csv("PreprocessedBank.csv").drop(['Unnamed: 0', 'unknown'],axis=1)
df.head()

#%%
# Tách X và y
X = df.drop(columns=["deposit"])
y = df["deposit"]

# Chuẩn hóa X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tạo DataFrame chuẩn hóa (nếu cần quan sát)
df_feat = pd.DataFrame(X_scaled, columns=X.columns)

# 30% Data is set aside for tesing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=3)
#%%

"""## KNN"""
#Training KNN
knn = KNeighborsClassifier(n_neighbors=9)
# Ensure y_train is of integer type
knn.fit(X_train, y_train.astype(int))

# Finding Accuracy, AUC, False positive rate, True positive rate, confusion matrix and classificatio report
pred=knn.predict(X_test)
y_pred_prob = knn.predict_proba(X_test)
aucScoreKNN = roc_auc_score(y_test,  y_pred_prob[:,1])
fprKNN, tprKNN, thresholds = roc_curve(y_test, y_pred_prob[:,1] )
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print("AUC score for KNN is ",aucScoreKNN)
print("Test Accuracy score for KNN is ",accuracy_score(y_test, pred))
predT=knn.predict(X_train)
print("Train Accuracy score for KNN is ",accuracy_score(y_train, predT))

# Training KNN for different odd values of k to find maximum Recall

knn = KNeighborsClassifier()
recall_rate=[]
for i in range(1,40,2):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,X_train,y_train,cv=10,scoring='recall')
    recall_rate.append(score.mean())
print(recall_rate)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
print("Test Recall score for KNN is ",recall_score(y_test, pred))
predT=knn.predict(X_train)
print("Train Recall score for KNN is ",recall_score(y_train, predT))
#%%
import joblib
joblib.dump(knn, "knn.pkl")
#%%

"""## SVM: Support Vector Machine"""

#Load the dataset
df = pd.read_csv('PreprocessedBank.csv')
#Khởi tạo mô hình
svm = SVC(C=3, class_weight='balanced', probability=True, random_state=3)
#Train SVM
svm.fit(X_train, y_train)

pred = svm.predict(X_test)
accSVM = accuracy_score(y_test, pred)
y_pred_prob = svm.predict_proba(X_test)
aucScoreSVM = roc_auc_score(y_test,  y_pred_prob[:,1])
fprSVM, tprSVM, thresholds = roc_curve(y_test, y_pred_prob[:,1] )
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print("AUC score for SVM is ",aucScoreSVM)
print("Test Accuracy score for SVM is ",accuracy_score(y_test, pred))
predT= svm.predict(X_train)
print("Train Accuracy score for SVM is ",accuracy_score(y_train, predT))

# Training SVM for recall
from sklearn.metrics import recall_score

# Predict for test and train
y_pred_test = svm.predict(X_test)
y_pred_train = svm.predict(X_train)

# Recall on test set
recall_test = recall_score(y_test, y_pred_test)

# Recall on train set
recall_train = recall_score(y_train, y_pred_train)

# In kết quả
print("Test Recall Score for SVM is:", recall_test)
print("Train Recall Score for SVM is:", recall_train)
#%%
import joblib
joblib.dump(svm, "svm.pkl")
#%%
# LR
#Logistic regression with polynomial features works better with normalozation instead of standardization so read file again

df = pd.read_csv("PreprocessedBank.csv").drop(['Unnamed: 0'],axis=1)
dfX=df.drop('deposit',axis=1)
df.head()

# Minmaxscaler is used to normalise data

scaler = MinMaxScaler()
bankMM = scaler.fit_transform(dfX)
bankMM = pd.DataFrame(bankMM, columns=dfX.columns)
bankMM.head()

# train test split (70:30)
X_train,X_test,y_train,y_test=train_test_split(bankMM,df['deposit'],test_size=0.30, random_state=3)

# creating polynomial features with degree 2
poly2 = PolynomialFeatures(degree=2)
X_trainP=poly2.fit_transform(X_train)

# Hyperparameters
param_grid = {'penalty' : ['l1', 'l2'], 'C' : [0.001,0.01,0.1,1,5,25]    }

#GridserchCV tries all possible combinations of hyperparameters to find best accuracy

logModel = LogisticRegression()
clf = GridSearchCV(logModel, param_grid = param_grid,scoring='accuracy',verbose=True, cv = 5,n_jobs=-1 )
best_clf = clf.fit(X_trainP,y_train)

# create polynomial features with degree 2 for test data
X_testP=poly2.fit_transform(X_test)

# Training Logistic Regression with polynomial features with degree 2
# Finding Accuracy, AUC, False positive rate, True positive rate, confusion matrix and classificatio report
# get best parameters for retraining

pred = best_clf.predict(X_testP)
accLRP2 = accuracy_score(y_test, pred)
y_pred_prob = best_clf.predict_proba(X_testP)
aucScoreLRP2 = roc_auc_score(y_test,  y_pred_prob[:,1])
fprLRP2, tprLRP2, thresholds = roc_curve(y_test, y_pred_prob[:,1] )
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print("AUC score for LR Poly2 is ",aucScoreLRP2)
print("Test Accuracy score for LR Poly2 is ",accLRP2)
predT=best_clf.predict(X_trainP)
print("Train Accuracy score for LR Poly2 is ",accuracy_score(y_train, predT))
print("Best parameters for LR are ",best_clf.best_params_)

logModel = LogisticRegression()
clfR = GridSearchCV(logModel, param_grid = param_grid,scoring='recall',verbose=True, cv = 5,n_jobs=-1 )
best_clfR = clfR.fit(X_trainP,y_train)

# Training Logistic Regression for recall

predR = best_clfR.predict(X_testP)
predRT=best_clfR.predict(X_trainP)
recallLRP2=recall_score(y_test, predR)
print("Test Recall score for LR with polynomial features degree 2  is ",recallLRP2)
print("Train recall score for LR with polynomial features degree 2 is ",recall_score(y_train, predRT))
print("Best parameters for recall of LR with polynomial features degree 2 are ",best_clfR.best_params_)
#%%
import joblib
joblib.dump(best_clf, "lr.pkl")
joblib.dump(poly2, 'poly2_transform.pkl')
#%%

# XGBoost
params = {
            'eta': np.arange(0.1, 0.26, 0.05),
            'min_child_weight': np.arange(1, 5, 0.5).tolist(),
            'gamma': [5],
            'subsample': np.arange(0.5, 1.0, 0.11).tolist(),
            'colsample_bytree': np.arange(0.5, 1.0, 0.11).tolist()
        }

xgb_model = XGBClassifier(objective = "binary:logistic")
skf = StratifiedKFold(n_splits=10, shuffle = True)
clf = GridSearchCV(xgb_model, param_grid = params, scoring = 'accuracy',cv = skf.split(X_train, y_train),verbose=True)
best_clf = clf.fit(X_train,y_train)
#%%
# Training XGBOOST
# Finding Accuracy, AUC, False positive rate, True positive rate, confusion matrix and classificatio report
# get best parameters for retraining

pred = best_clf.predict(X_test)
accXGBOOST = accuracy_score(y_test, pred)
y_pred_prob = best_clf.predict_proba(X_test)
aucScoreXGBOOST = roc_auc_score(y_test,  y_pred_prob[:,1])
fprXGBOOST, tprXGBOOST, thresholds = roc_curve(y_test, y_pred_prob[:,1] )
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print("AUC score for XGBOOST is ",aucScoreXGBOOST)
print("Test Accuracy score for XGBOOST is ",accuracy_score(y_test, pred))
predT=best_clf.predict(X_train)
print("Train Accuracy score for XGBOOST is ",accuracy_score(y_train, predT))
print("Best parameters for XGBOOST are ",best_clf.best_params_)

#%%
#Gridsearchcv And Training XGBBOST for recall
clfR = GridSearchCV(estimator=xgb_model, param_grid = params, scoring = 'recall',cv = skf,verbose=True)
best_clfR = clfR.fit(X_train,y_train)

predR = best_clfR.predict(X_test)
predRT=best_clfR.predict(X_train)
recallXGBOOST=recall_score(y_test, predR)
print("Test Recall score for XGBOOST is ",recallXGBOOST)
print("Train recall score for XGBOOST is ",recall_score(y_train, predRT))
print("Best parameters for recall of XGBOOST are ",best_clfR.best_params_)
#%%
import joblib
joblib.dump(best_clfR, "XGBoost.pkl")

#%%
import gdown
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Link file Google Drive
file_id = '1sAO6Dq7Lpj78qe8w_OmZNjs9h9L97mhs'
url = f'https://drive.google.com/uc?id={file_id}'

# Tải file về
gdown.download(url, 'PreprocessedBank.csv', quiet=False)

# Đọc dữ liệu và loại bỏ các cột không cần thiết
df = pd.read_csv("PreprocessedBank.csv").drop(['Unnamed: 0', 'unknown'], axis=1)

# Lưu lại tên các cột
feature_names = df.columns
with open("feature_names.txt", "w") as f:
    for col in feature_names:
        f.write(col + "\n")

# Phân tách dữ liệu thành X, y
X = df.drop("deposit", axis=1)
y = df["deposit"]

# Chia tập train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=3)

# Fit scaler
scaler = StandardScaler()
scaler.fit(X_train)

# Lưu scaler
joblib.dump(scaler, 'scaler.pkl')

df.columns
# %%
