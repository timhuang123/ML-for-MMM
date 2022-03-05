# load and preprocess data
import pandas as pd
import numpy as np
data=pd.read_excel(open('/content/Book3-3.xlsx','rb'),'CO2-CH4')
data=data.iloc[:,1:]
data.head()

numerical_data=data.drop(['filler type.1','matrix type'],axis=1).values

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

## obtain correlation matrix
# correlation matrix for co2-ch4
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

# Generate a large random dataset
rs = np.random.RandomState(33)
d = df

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool),1)


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18,20))
ax.tick_params(axis='both', which='major', labelsize=20)
#ax.set_xlabel('x', fontweight='bold')

# Generate a custom diverging colormap
#cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.set(font_scale=2.5)
sns.heatmap(corr, mask=mask, cmap='seismic', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5},vmax=1,vmin=-1)

#plt.xticks(rotation=60)
plt.xticks(weight = 'bold')
#plt.yticks(rotation=20)
plt.yticks(weight = 'bold')
f.savefig("corr_co2_ch4.png",dpi=600)
