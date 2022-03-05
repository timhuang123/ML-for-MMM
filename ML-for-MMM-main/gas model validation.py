import pandas as pd
import numpy as np
data=pd.read_excel(open('/content/Book3-3-add-exp-data.xlsx','rb'),'CO2-CH4')
data=data.iloc[:-8,1:]
data.tail()

numerical_data=data.drop(['filler type.1','matrix type'],axis=1).values
numerical_data.shape
data_knn_imputed = KNN(k=5, verbose=0).fit_transform(numerical_data)
data_knn_imputed.shape

columns_name=data.drop(['filler type.1','matrix type'],axis=1).columns
columns_name

for col in ['filler type.1', 'matrix type']:
    data[col] = data[col].astype('category')

df=pd.DataFrame(data_knn_imputed)
df.columns=columns_name

df1=pd.concat([data.iloc[:,:2],df],axis=1)
df1.head(3)


import pandas as pd
data_validation=pd.read_excel(open('/content/gas_validation_pim.xlsx','rb'),'Sheet1').iloc[:,:]
data_validation





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



df1=pd.concat([data.iloc[:,:2],df],axis=1)
df1.shape

from sklearn.utils import shuffle
df1 = shuffle(df1)

df1.columns=['filler type.1','matrix type','$\mathregular{M_s}$','BET', '$\mathregular{S_c}$', '$\mathregular{S_a}$',
       '$\mathregular{LA_m}$', 'TM', 'T', 'P',
       '$\mathregular{CP_{CH4}}$', '$\mathregular{CS_{CO2/CH4}}$',
       '$\mathregular{RP_{CO2}}$', '$\mathregular{RS_{CO2/CH4}}$']


X, y = df1.iloc[:,0:12],df1.iloc[:,12]

categorical_columns = ['filler type.1', 'matrix type']
numerical_columns = list(columns_name[:-2])

X = X[categorical_columns + numerical_columns]

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=45)

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
est.fit(X,y)

# save model
from joblib import dump, load
dump(est, 'model_est_perm_retrained_with_exp_data_v1.joblib')

#load model
model_est_permeability = load('/content/model_est_perm_retrained_with_exp_data_v1.joblib') 
perm_pred=model_est_permeability.predict(data_validation.iloc[:4,:12])
perm_pred


