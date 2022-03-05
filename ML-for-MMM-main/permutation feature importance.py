# load and preprocess data
import pandas as pd
import numpy as np
data=pd.read_excel(open('/content/Data-MOFS.csv','rb'),'Data-MOFS')
data=data.iloc[:,:]
data.head()

numerical_data=data.drop(['filler type','matrix type'],axis=1).values

from fancyimpute import KNN
data_knn_imputed = KNN(k=5, verbose=0).fit_transform(numerical_data)
data_knn_imputed.shape


#descriptive feature names
columns_name_des= ['Filler size', 'BET', 'Pore size',
       'Aperture size', 'Loading', 'Thickness', 'Temperature', 'Pressure',
       'Control permeability', 'Control selectivity',
       'Relative permeability', 'Relative selectivity']

for col in ['filler type.1', 'matrix type']:
    data[col] = data[col].astype('category')

df=pd.DataFrame(data_knn_imputed)
df.columns=columns_name_des


data2=pd.concat([data.iloc[:,:2],df],axis=1)
data2.head()

# build model and obtain permutation feature importance
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

X, y = data2.iloc[:,0:12],data2.iloc[:,13]

categorical_columns = ['filler type.1', 'matrix type']
numerical_columns = list(columns_name_br[:-2])

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
    ('regressor', RandomForestRegressor(random_state=42))
])
est.fit(X,y)

from sklearn.inspection import permutation_importance
r = permutation_importance(est, X_train, y_train,
                           n_repeats=30,
                          random_state=0)
    
