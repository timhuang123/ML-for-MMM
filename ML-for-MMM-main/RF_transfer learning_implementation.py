# import basic library
from __future__ import division, print_function
import numpy as np
import math
import progressbar

# Import helper functions
from ml_lib.utils import divide_on_feature, train_test_split, get_random_subsets, normalize
from ml_lib.utils import accuracy_score, calculate_entropy
from ml_lib.utils.misc import bar_widgets
from ml_lib.utils import Plot

# for regression tree
from ml_lib.supervised_learning import RegressionTree

## implement rf model
class RandomForest_r():
    """Random Forest classifier. Uses a collection of classification trees that
    trains on random subsets of the data using a random subsets of the features.

    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    max_features: int
        The maximum number of features that the classification trees are allowed to
        use.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_gain: float
        The minimum impurity required to split the tree further. 
    max_depth: int
        The maximum depth of a tree.
    """
    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2,
                 min_gain=0, max_depth=float("inf")):
        self.n_estimators = n_estimators    # Number of trees
        self.max_features = max_features    # Maxmimum number of features per tree
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain            # Minimum information gain req. to continue
        self.max_depth = max_depth          # Maximum depth for tree
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

        # Initialize decision trees
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(
                RegressionTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_gain,
                    max_depth=self.max_depth))

    def fit(self, X, y):
        n_features = np.shape(X)[1]
        # If max_features have not been defined => select it as
        # sqrt(n_features)
        if not self.max_features:
            self.max_features = int(math.sqrt(n_features))

        # Choose one random subset of the data for each tree
        subsets = get_random_subsets(X, y, self.n_estimators)

        for i in self.progressbar(range(self.n_estimators)):
            X_subset, y_subset = subsets[i]
            # Feature bagging (select random subsets of the features)
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            # Save the indices of the features for prediction
            self.trees[i].feature_indices = idx
            # Choose the features corresponding to the indices
            X_subset = X_subset[:, idx]
            # Fit the tree to the data
            self.trees[i].fit(X_subset, y_subset)

    def predict(self, X):
        y_preds = np.empty((X.shape[0], len(self.trees)))
        # Let each tree make a prediction on the data
        for i, tree in enumerate(self.trees):
            # Indices of the features that the tree has trained on
            idx = tree.feature_indices
            # Make a prediction based on those features
            prediction = tree.predict(X[:, idx])
            y_preds[:, i] = prediction
            
        y_pred = []
        # For each sample
        for sample_predictions in y_preds:
            # Select the most common class prediction
            y_pred.append(np.mean(sample_predictions,axis=None))
        return y_pred

#########
# training for co2-ch4
from fancyimpute import KNN
import pandas as pd

data1=pd.read_excel(open('/content/Book3-3.xlsx','rb'),'CO2-CH4').iloc[:,3:]

numerical_data1=data1.values

data_knn_imputed1 = KNN(k=5, verbose=0).fit_transform(numerical_data1)
data_knn_imputed1.shape

data2=data_knn_imputed1

X=data2[:,0:-2]
##-2 -> perm
##-1 -> sel
y=data2[:,-2]


#use train test split to validate
from sklearn.model_selection import train_test_split
clf = RandomForest_r(max_depth=12,max_features=9)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

clf.fit(X, y)

## extract paths of decision trees
def extract_path(index):
  temp0=clf.trees[index]
  helper=[]
  idx=[]
  for k in range(len(X)):
    temp0.predict_value(X[k])
    a=temp0.res
    helper.append(a)
    idx.append(len(a))
  
  idx1=[(idx[i],idx[i+1]) for i in range(len(idx)-1)]
  idx1=[(0,idx[0])]+idx1
  arr=[]
  for i in range(len(X)):
    arr.append(helper[645][idx1[i][0]:idx1[i][1]])

  arr_str=[]
  for i in range(len(X)):
    arr_str.append([[str(i) for i in sublist] for sublist in arr[i]])

  arr_flat=[]
  for i in range(len(X)):
    arr_flat.append(['-'.join(sublist) for sublist in arr_str[i]])

  import pickle
  with open('./arr_flat/arr_flat_%d.pkl'%(index), 'wb') as f:
    pickle.dump(arr_flat, f)

# to extract all

for i in range(100):
  extract_path(i)


## compile all path without duplicate
## compile all paths
import pickle
paths=[]
# load data
for i in range(100):
  with open('./arr_flat_co2-ch4/arr_flat_%d.pkl'%(i), 'rb') as f:
    arr_flat_p = pickle.load(f)
  
  #arr_flat_p
  list_=[]
  for k,v in enumerate(arr_flat_p):
    if v not in list_:
      list_.append(v)
  for i in list_:
    if i not in paths:
      paths.append(i)
#
  
# load data
datapoint_paths={i:[] for i in range(len(paths))}
for i in range(100):
  with open('./arr_flat_co2-ch4/arr_flat_%d.pkl'%(i), 'rb') as f:
    arr_flat_p = pickle.load(f)
  
  for k,item in enumerate(arr_flat_p):
    idx=paths.index(item)
    datapoint_paths[idx].append(k)

## 
with open('./paths_co2-ch4.pkl', 'wb') as f:
  pickle.dump(paths, f)
## 
with open('./datapoint_paths_co2-ch4.pkl', 'wb') as f:
  pickle.dump(datapoint_paths, f)

##training for co2-n2
import pandas as pd

data1=pd.read_excel(open('/content/CO2-N2 - 1.xlsx','rb'),'CO2-N2').iloc[:,2:]

numerical_data1=data1.values

data_knn_imputed1 = KNN(k=5, verbose=0).fit_transform(numerical_data1)
data_knn_imputed1.shape

data2=data_knn_imputed1

## prepare data
X=data2[:,0:-2]
#-2 ->perm
#-1 ->sel
y=data2[:,-2]
  
#use train test split to validate
from sklearn.model_selection import train_test_split
clf = RandomForest_r(max_depth=9,max_features=9)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

clf.fit(X_train, y_train)

##
def extract_path(index):
  temp0=clf.trees[index]
  helper=[]
  idx=[]
  for k in range(len(X_train)):
    temp0.predict_value(X_train[k])
    a=temp0.res
    helper.append(a)
    idx.append(len(a))
  
  idx1=[(idx[i],idx[i+1]) for i in range(len(idx)-1)]
  idx1=[(0,idx[0])]+idx1
  arr=[]
  for i in range(len(X_train)):
    arr.append(helper[309][idx1[i][0]:idx1[i][1]])

  arr_str=[]
  for i in range(len(X_train)):
    arr_str.append([[str(i) for i in sublist] for sublist in arr[i]])

  arr_flat=[]
  for i in range(len(X_train)):
    arr_flat.append(['-'.join(sublist) for sublist in arr_str[i]])

  import pickle
  with open('./arr_flat_co2-n2/arr_flat_%d.pkl'%(index), 'wb') as f:
    pickle.dump(arr_flat, f)

# to extract all paths
for i in range(100):
  extract_path(i)

##compile all paths of co2-n2
## compile all path
import pickle
paths=[]
# load data
for i in range(100):
  with open('./arr_flat_co2-n2/arr_flat_%d.pkl'%(i), 'rb') as f:
    arr_flat_p = pickle.load(f)
  
  #arr_flat_p
  list_=[]
  for k,v in enumerate(arr_flat_p):
    if v not in list_:
      list_.append(v)
  ##remove duplicate
  for item in list_:
    if item not in paths:
      paths.append(item)
#paths

# load data
datapoint_paths={i:[] for i in range(len(paths))}
for i in range(100):
  with open('./arr_flat_co2-n2/arr_flat_%d.pkl'%(i), 'rb') as f:
    arr_flat_p = pickle.load(f)
  
  for k,item in enumerate(arr_flat_p):

      idx=paths.index(item)
      
      datapoint_paths[idx].append(k)

  
## save data
with open('./paths_co2-n2.pkl', 'wb') as f:
  pickle.dump(paths, f)

with open('./datapoint_paths_co2-n2.pkl', 'wb') as f:
  pickle.dump(datapoint_paths, f)


##compare and find the same path for co2-ch4 and co2-n2
with open('./paths_co2-ch4.pkl', 'rb') as f:
  paths_co2_ch4=pickle.load(f)

with open('./paths_co2-n2.pkl', 'rb') as f:
  paths_co2_n2=pickle.load(f)

## find corresponding data
def check_subpath(i,j):
  if len(i)<=len(j):
    for k,v in enumerate(i):
      if v!=j[k]: return False
    return True
  
corresponding_data=[]
for k,path1 in enumerate(paths_co2_ch4):
  for j,path2 in enumerate(paths_co2_n2):
    if check_subpath(path1,path2):
      #print(0)
      corresponding_data.append([k,j])

#obtain matched data points

import pickle
with open('./datapoint_paths_co2-ch4.pkl', 'rb') as f:
  dpaths_co2_ch4=pickle.load(f)

with open('./datapoint_paths_co2-n2.pkl', 'rb') as f:
  dpaths_co2_n2=pickle.load(f)

## find and save matched data points
matched_datapoints=[]
for k,v in corresponding_data:
  matched_datapoints.append((dpaths_co2_ch4[k],dpaths_co2_n2[v]))

#matched_datapoints
  
## compute similarity of data point distribution

import pandas as pd

data_co2_ch4=pd.read_excel(open('/content/Book3-3.xlsx','rb'),'CO2-CH4').iloc[:,3:]

numerical_data1=data_co2_ch4.values

data_knn_imputed_co2_ch4 = KNN(k=5, verbose=0).fit_transform(numerical_data1)

## remove duplicate

len(matched_datapoints)
matched_datapoints_remove_duplicate=[]
for i in range(35):
  matched_datapoints_remove_duplicate.append([list(set(matched_datapoints[i][0])), list(set(matched_datapoints[i][1]))])

##
cnt=0
for i in range(35):
  cnt+=1
  temp=stats.kstest(data_knn_imputed_co2_ch4[matched_datapoints_remove_duplicate[i][0],10], data_knn_imputed_co2_n2[matched_datapoints_remove_duplicate[i][1],10])
  print(temp)
cnt



import pandas as pd

data_co2_n2=pd.read_excel(open('/content/CO2-N2 - 1.xlsx','rb'),'CO2-N2').iloc[:,2:]

numerical_data1=data_co2_n2.values

data_knn_imputed_co2_n2 = KNN(k=5, verbose=0).fit_transform(numerical_data1)


# use lasso to obtain sparse mapping function
mean_vector=[]

for i in range(35):
  helper1=data_knn_imputed_co2_ch4[matched_datapoints_remove_duplicate[i][0],0:10]
  helper1_mean=np.mean(helper1,axis=0)

  helper2=data_knn_imputed_co2_n2[matched_datapoints_remove_duplicate[i][1],0:10]
  helper2_mean=np.mean(helper2,axis=0)

  mean_vector.append([helper1_mean, helper2_mean])

len(mean_vector)

# regr is a mapping function which transform target feature into a linear combination of source features
for j in range(10):
  res=[]
  for i in range(35):
    res.append([mean_vector[i][0][j]]+list(mean_vector[i][1]))

  # prepare data
  temp=np.array(res)
  X=temp[:,1:]
  y=temp[:,0]

  # training model
  regr = linear_model.Lasso(alpha=0.1)
  regr.fit(X,y)

  # save the model to disk
  #filename = 'finalized_model.sav'
  pickle.dump(regr, open('regr_perm_%d'%(j), 'wb'))
## this is end of script
