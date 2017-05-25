import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# load data
df_train = pd.read_csv('data.csv')

# check the decoration
print(df_train.columns)

# descriptive statistics summary
print(df_train['SalePrice'].describe())

# histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm)
sns.plt.show()


# saleprice correlation matrix
corrmat = df_train.corr()
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
sns.plt.show()

# box-plot grlivarea/saleprice
var = 'LandContour'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
print(pd.unique(data['LandContour'].values))
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
sns.plt.show()

# one way ANOVA test
mod = ols('SalePrice ~ LandContour', data=data).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
print(esq_sm)

# missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

# applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])
print(df_train['SalePrice'].describe())

# transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm)
sns.plt.show()







