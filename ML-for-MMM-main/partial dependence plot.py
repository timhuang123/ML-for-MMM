import pandas as pd
import numpy as np
data=pd.read_excel(open('/content/Book3-3.xlsx','rb'),'CO2-CH4')
data=data.iloc[:,1:]
data.head()

numerical_data=data.drop(['filler type.1','matrix type'],axis=1).values
numerical_data.shape

from fancyimpute import KNN
data_knn_imputed = KNN(k=5, verbose=0).fit_transform(numerical_data)
data_knn_imputed.shape

columns_name=data.drop(['filler type.1','matrix type'],axis=1).columns
columns_name

#描述型feature名字 with unit
columns_name_des_unit= ['Filler size/nm', '$\mathregular{BET/m^2 g^{-1}}$', 'Pore size/Å',
       'Aperture size/Å', 'Loading/wt%', 'Thickness/μm', 'Temperature/℃', 'Pressure/bar',
       'Control permeability', 'Control selectivity',
       'Relative permeability', 'Relative selectivity']

for col in ['filler type.1', 'matrix type']:
    data[col] = data[col].astype('category')

# transform loading to %
multi=np.array([1,1,1,1,100,1,1,1,1,1,1,1])

data_knn_imputed1=np.multiply(data_knn_imputed, multi)

df=pd.DataFrame(data_knn_imputed1)
df.columns=columns_name_des_unit
df.shape

df1=pd.concat([data.iloc[:,:2],df],axis=1)
df1.head(5)

# shuffle data
from sklearn.utils import shuffle
df1 = shuffle(df1)

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

# build model
X, y = df1.iloc[:,0:12],df1.iloc[:,13]

categorical_columns = ['filler type.1', 'matrix type']
numerical_columns = list(columns_name_des_unit[:-2])

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

# obtain permutation value for each variable
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline

from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from sklearn.experimental import enable_hist_gradient_boosting  # noqa


# 1D pdp
for i in range(10):

  features = columns_name_des_unit[:10][i:i+1]

  plot_partial_dependence(est, X_train, features,
                          n_jobs=3, grid_resolution=20,line_kw={"color": "black",'linewidth':3})

  import matplotlib.pyplot as plt
  fig = plt.gcf()
  plt.rcParams["figure.figsize"] = (7,7)

  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)

  plt.rcParams['axes.labelsize'] = 22
  plt.savefig('pdp_%d.png'%(i),dpi=600)
  fig.suptitle('Partial dependence of permeability on features')
  fig.subplots_adjust(hspace=0.6)


