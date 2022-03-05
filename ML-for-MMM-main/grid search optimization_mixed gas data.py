from fancyimpute import KNN

import pandas as pd
data1=pd.read_excel(open('/content/Mix gas data-GJ 27 July.xlsx','rb'),'mixed').iloc[:,1:-2]

numerical_data1=data1.drop(['Polymer Type', 'MOF Type'],axis=1).values

data_knn_imputed1 = KNN(k=5, verbose=0).fit_transform(numerical_data1)
data_knn_imputed1.shape

for col in ['Polymer Type', 'MOF Type']:
    data1[col] = data1[col].astype('category')

data2=pd.concat([data1.iloc[:,:1],data1.iloc[:,6:7],pd.DataFrame(data_knn_imputed1).iloc[:,:]],axis=1)
#data2.columns=data1.columns
data2.shape
data2.head()


#import library
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

categorical_columns = list(data2.columns[:2])
numerical_columns = list(data2.columns[2:-2])

X = X[categorical_columns + numerical_columns]

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=44) #random_state=44 is good

X=pd.concat([X_train,X_test],axis=0)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X['MOF Type'] = labelencoder.fit_transform(X['MOF Type'])
X['Polymer Type'] = labelencoder.fit_transform(X['Polymer Type'])

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 2)

X_train=X.iloc[:113,:]
from sklearn.model_selection import GridSearchCV
param_grid = {'max_features': [5,9,12],'n_estimators': [100, 200, 300]}
regr = GridSearchCV(rf, param_grid,cv = 3, n_jobs = -1, verbose = 2,scoring='r2')
regr.fit(X_train, y_train)

best_model = regr.best_estimator_
X_test=X.iloc[113:,:]
pred=best_model.predict(X_test)


# report the score on testing dataset
from sklearn.metrics import r2_score
r2_score(y_test,pred)


