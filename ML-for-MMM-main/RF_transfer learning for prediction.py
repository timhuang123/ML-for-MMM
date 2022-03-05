#load and preprocess data
from fancyimpute import KNN
# load co2-n2 data
import pandas as pd

data1=pd.read_excel(open('/content/CO2-N2 - 2.xlsx','rb'),'CO2-N2')
data1.head()

numerical_data1=data1.drop(['filler type','matrix type'],axis=1).values

data_knn_imputed1 = KNN(k=5, verbose=0).fit_transform(numerical_data1)
data_knn_imputed1.shape

for col in ['filler type', 'matrix type']:
    data1[col] = data1[col].astype('category')

data2=pd.concat([data1.iloc[:,:2],pd.DataFrame(data_knn_imputed1)],axis=1)
data2.columns=data1.columns
data2.shape
data2.head()

# make direct prediction
print(__doc__)
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 12->perm
# 13->sel
X, y = data2.iloc[:,0:12],data2.iloc[:,13]

categorical_columns = ['filler type', 'matrix type']
numerical_columns = ['filler size/nm', 'BET', 'cage size/A',
       'aperture size/A', 'loading amout', 'thickness', 'T/℃', 'P/bar',
       'Control N2 permeability', 'Control CO2/N2 selectivity']

X = X[categorical_columns + numerical_columns]

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=44) #initial random state is 45

categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

preprocessing = ColumnTransformer(
    [('cat', categorical_pipe, categorical_columns),
     ('num', numerical_pipe, numerical_columns)])

est = Pipeline([
    ('preprocess', preprocessing),
    ('regressor', RandomForestRegressor(n_estimators=300,max_features=12,random_state=2))
])
est.fit(X_train,y_train)

perm_test_pred=est.predict(X_test)
perm_test_pred

# sort the values and not keep outlier 
test_pred_zipped=list(zip(y_test,perm_test_pred))
test_pred_sorted=sorted(d_zip,key=lambda x:x[0])[:-2]

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(test_pred_sorted[0],test_pred_sorted[1]))
rms

from sklearn.metrics import r2_score
r2_score(test_pred_sorted[0],test_pred_sorted[1])

## RF transfer learning prediction

X_test_num=X_test.iloc[:,2:]
X_test_num.shape

# load the model from disk
import pickle
transformed_feature=[]
for i in range(0,2,1):
  loaded_model = pickle.load(open('./rf_transfer_perm/regr_model_perm/regr_perm_%d'%(i), 'rb'))
  result = loaded_model.predict(X_test_num)
  transformed_feature.append(list(result))

transformed_target=np.array(transformed_feature).T
transformed_target.shape

X_test_trans=pd.DataFrame(np.hstack((X_test.iloc[:,:2].values,transformed_target,X_test_num.iloc[:,2:])))
X_test_trans.columns= ['filler type.1', 'matrix type', 'filler size/nm', 'BET',
       'cage size/A', 'aperture size/A', 'loading amout', 'thickness/μm',
       'T/℃', 'P/bar', 'Control CH4 permeability',
       'Control CO2/CH4 selectivity']

from joblib import dump, load
#dump(est, 'model_est_permeability.joblib') 

est = load('/content/model_est_selectivity.joblib') 
#data2.head(3)

perm_pred=est.predict(X_test_trans)
perm_pred
# sort the values and not keep outlier 
test_pred_zipped=list(zip(y_test,perm_pred))
test_pred_sorted=sorted(d_zip,key=lambda x:x[0])[:-2]

r2_score(test_pred_sorted[0],test_pred_sorted[1])
