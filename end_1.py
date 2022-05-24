
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
from matplotlib import pyplot
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from category_encoders.one_hot import OneHotEncoder
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest,chi2
import scipy.stats as stats
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

##
pd.set_option('display.max_columns',10)
df = pd.read_csv('train_set.csv')
test = pd.read_csv('test_set.csv')
sample_sub = pd.read_csv('sub_example.csv')
### drop ##
df = df.drop('id', axis=1)
df = df.drop('Driving_License', axis=1)

# df = df.drop('Driving_License', axis=1)
test = test.drop('id', axis=1)
test = test.drop('Driving_License', axis=1)

# print(df1.isnull().sum())

##  OneHotEncoder()
import category_encoders as ce
enc = ce.BinaryEncoder()
cols_encoding = df.select_dtypes(include='object').columns
cols_encoding1 = test.select_dtypes(include='object').columns
ohe1 = OneHotEncoder(cols=cols_encoding1)
ohe = OneHotEncoder(cols=cols_encoding)
df = ohe.fit_transform(df)
test = ohe1.fit_transform(test)

### normalize
data = preprocessing.normalize(df)

### Feature Selection
data = df.drop('Response', axis=1)

Target = df['Response']

# chi2_feature = SelectKBest(chi2,k = 2 )
# X_best_feature = chi2_feature.fit_transform(data,Target)
#
# print('orignal ', data.shape[1])
# print('reduce',X_best_feature.shape[1])

# X_train1, X_test, y_train1, y_test = train_test_split(data,Target,test_size=0.3,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(data,Target,test_size=0.1,random_state=0)

# X_train ,y_train = SMOTEENN().fit_resample(X_train1, y_train1)

def correlation(data, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i , j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, .8)
print('correlated features: ', len(set(corr_features)) )
print(corr_features)
data.drop(labels=corr_features, axis=1, inplace=True)
test.drop(labels=corr_features, axis=1, inplace=True)
# data.drop(labels='Annual_Premium', axis=1, inplace=True)
# test.drop(labels='Annual_Premium', axis=1, inplace=True)

print(data.shape, test.shape)
### imbalanced data
#

RandomizedSearchCV
### Train ##

### save SCV

params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
}
# xgboost.DMatrix
# rf = xgboost.XGBClassifier(learning_rate= 0.2, n_estimators =  100, objective =  'binary:logistic',tree_method='gpu_hist',param_grid={'C': 1, 'kernel': 'linear'},class_wegiht='balanced')
# random_search=RandomizedSearchCV(rf,param_distributions=params,n_iter=10,scoring='roc_auc',n_jobs=-1,cv=3,verbose=3)
eval_set = [(X_train, y_train), (X_test, y_test)]
X_train_ = np.concatenate((X_train[::2], X_train[::2]), axis=0)
y_train_ = np.concatenate((y_train[::2], y_train[::2]), axis=0)
w_train = 0.5 * np.ones_like(y_train_)
rf = XGBClassifier()
rf.fit(X_train_, y_train_, w_train)
rf.fit(X_train, y_train,eval_metric=["error", "auc"], eval_set=eval_set, verbose=True,early_stopping_rounds=10)
print('Train set')
pred = rf.predict_proba(X_train)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = rf.predict_proba(X_test)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
rf.fit(data, Target)
predictions = rf.predict(test)
print(predictions)
results = rf.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()
sample_sub['Response'] = predictions
sample_sub.to_csv('fixed_output__02.csv', index=False)

df = pd.read_csv('fixed_output__02.csv')
a = df.Response == 1
b = df.Response == 0
print(a.sum())
print(b.sum())


# 16197
# 16850
# 20199
