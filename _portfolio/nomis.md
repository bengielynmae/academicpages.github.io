```python
import numpy as np
import matplotlib.pyplot as plt
import pylab as plot
%matplotlib inline
import pandas as pd
import seaborn as sns
from collections import Counter

#classifiers/regressions
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, LinearRegression
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, Imputer
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning

import warnings
from warnings import simplefilter
simplefilter('ignore')

from tqdm.autonotebook import tqdm
from functools import partial
```


```python
plt.style.use('https://gist.githubusercontent.com/lpsy/e81ff2c0decddc9c6dfeb2fcffe37708/raw/lsy_personal.mplstyle')
```


```python
df = pd.read_excel('NomisB.xlsx')
dummy = df
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tier</th>
      <th>FICO</th>
      <th>Approve Date</th>
      <th>Term</th>
      <th>Amount</th>
      <th>Previous Rate</th>
      <th>Car  Type</th>
      <th>Competition rate</th>
      <th>Outcome</th>
      <th>Rate</th>
      <th>Cost of Funds</th>
      <th>Partner Bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>743</td>
      <td>2002-07-01</td>
      <td>36</td>
      <td>19100.00</td>
      <td></td>
      <td>N</td>
      <td>4.95</td>
      <td>1</td>
      <td>4.85</td>
      <td>1.8388</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>752</td>
      <td>2002-07-01</td>
      <td>60</td>
      <td>36460.08</td>
      <td></td>
      <td>N</td>
      <td>5.65</td>
      <td>1</td>
      <td>5.49</td>
      <td>1.8388</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>778</td>
      <td>2002-07-01</td>
      <td>48</td>
      <td>12000.00</td>
      <td></td>
      <td>U</td>
      <td>5.85</td>
      <td>1</td>
      <td>5.85</td>
      <td>1.8388</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>724</td>
      <td>2002-07-01</td>
      <td>60</td>
      <td>19290.00</td>
      <td></td>
      <td>N</td>
      <td>5.65</td>
      <td>1</td>
      <td>5.39</td>
      <td>1.8388</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>700</td>
      <td>2002-07-01</td>
      <td>72</td>
      <td>24461.12</td>
      <td></td>
      <td>N</td>
      <td>6.25</td>
      <td>1</td>
      <td>6.99</td>
      <td>1.8388</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 208085 entries, 0 to 208084
    Data columns (total 12 columns):
    Tier                208085 non-null int64
    FICO                208085 non-null int64
    Approve Date        208085 non-null datetime64[ns]
    Term                208085 non-null int64
    Amount              208085 non-null float64
    Previous Rate       208077 non-null object
    Car  Type           208085 non-null object
    Competition rate    208085 non-null float64
    Outcome             208085 non-null int64
    Rate                208085 non-null float64
    Cost of Funds       208085 non-null float64
    Partner Bin         208085 non-null int64
    dtypes: datetime64[ns](1), float64(4), int64(5), object(2)
    memory usage: 19.1+ MB



```python
df.shape
```




    (208085, 12)




```python
df['Previous Rate'] = df[['Previous Rate']].convert_objects(convert_numeric=True).fillna(0)
```


```python

```


```python
# state_counts = Counter(df['Outcome'])
sns.countplot(x='Outcome', data=df)
# df_state = pd.DataFrame.from_dict(state_counts, orient='index')
# df_state.plot(kind='bar')

df_state = df['Outcome'].value_counts()
num=(df_state/df_state.sum())**2
print("Population per class:\n{}".format(df_state))
print("1.25 * Proportion Chance Criterion: {}%".format(1.25*100*num.sum()))
```

    Population per class:
    0    162298
    1     45787
    Name: Outcome, dtype: int64
    1.25 * Proportion Chance Criterion: 82.09441614132041%



![png](ML_Lab1_Nomis_files/ML_Lab1_Nomis_8_1.png)



```python
X_clean = df
```


```python
# df = pd.read_excel('NomisB.xlsx')
size_mapping = {
1: 3,
2: 2,
3: 1,
4: 0}
X_clean['Tier'] = X_clean['Tier'].map(size_mapping)
```


```python
X_clean['Partner Bin'] = X_clean['Partner Bin'].astype(object)
```


```python
X_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 208085 entries, 0 to 208084
    Data columns (total 12 columns):
    Tier                208085 non-null int64
    FICO                208085 non-null int64
    Approve Date        208085 non-null datetime64[ns]
    Term                208085 non-null int64
    Amount              208085 non-null float64
    Previous Rate       208085 non-null float64
    Car  Type           208085 non-null object
    Competition rate    208085 non-null float64
    Outcome             208085 non-null int64
    Rate                208085 non-null float64
    Cost of Funds       208085 non-null float64
    Partner Bin         208085 non-null object
    dtypes: datetime64[ns](1), float64(5), int64(4), object(2)
    memory usage: 19.1+ MB



```python
XX=pd.get_dummies(X_clean)
```


```python
X_clean.shape
```




    (208085, 12)




```python
XX.shape
```




    (208085, 16)




```python
XX.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 208085 entries, 0 to 208084
    Data columns (total 16 columns):
    Tier                208085 non-null int64
    FICO                208085 non-null int64
    Approve Date        208085 non-null datetime64[ns]
    Term                208085 non-null int64
    Amount              208085 non-null float64
    Previous Rate       208085 non-null float64
    Competition rate    208085 non-null float64
    Outcome             208085 non-null int64
    Rate                208085 non-null float64
    Cost of Funds       208085 non-null float64
    Car  Type_N         208085 non-null uint8
    Car  Type_R         208085 non-null uint8
    Car  Type_U         208085 non-null uint8
    Partner Bin_1       208085 non-null uint8
    Partner Bin_2       208085 non-null uint8
    Partner Bin_3       208085 non-null uint8
    dtypes: datetime64[ns](1), float64(5), int64(4), uint8(6)
    memory usage: 17.1 MB



```python
XX.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tier</th>
      <th>FICO</th>
      <th>Approve Date</th>
      <th>Term</th>
      <th>Amount</th>
      <th>Previous Rate</th>
      <th>Competition rate</th>
      <th>Outcome</th>
      <th>Rate</th>
      <th>Cost of Funds</th>
      <th>Car  Type_N</th>
      <th>Car  Type_R</th>
      <th>Car  Type_U</th>
      <th>Partner Bin_1</th>
      <th>Partner Bin_2</th>
      <th>Partner Bin_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>743</td>
      <td>2002-07-01</td>
      <td>36</td>
      <td>19100.00</td>
      <td>0.0</td>
      <td>4.95</td>
      <td>1</td>
      <td>4.85</td>
      <td>1.8388</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>752</td>
      <td>2002-07-01</td>
      <td>60</td>
      <td>36460.08</td>
      <td>0.0</td>
      <td>5.65</td>
      <td>1</td>
      <td>5.49</td>
      <td>1.8388</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>778</td>
      <td>2002-07-01</td>
      <td>48</td>
      <td>12000.00</td>
      <td>0.0</td>
      <td>5.85</td>
      <td>1</td>
      <td>5.85</td>
      <td>1.8388</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>724</td>
      <td>2002-07-01</td>
      <td>60</td>
      <td>19290.00</td>
      <td>0.0</td>
      <td>5.65</td>
      <td>1</td>
      <td>5.39</td>
      <td>1.8388</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>700</td>
      <td>2002-07-01</td>
      <td>72</td>
      <td>24461.12</td>
      <td>0.0</td>
      <td>6.25</td>
      <td>1</td>
      <td>6.99</td>
      <td>1.8388</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_clean.columns
```




    Index(['Tier', 'FICO', 'Approve Date', 'Term', 'Amount', 'Previous Rate',
           'Car  Type', 'Competition rate', 'Outcome', 'Rate', 'Cost of Funds',
           'Partner Bin'],
          dtype='object')




```python
XX.columns
```




    Index(['Tier', 'FICO', 'Approve Date', 'Term', 'Amount', 'Previous Rate',
           'Competition rate', 'Outcome', 'Rate', 'Cost of Funds', 'Car  Type_N',
           'Car  Type_R', 'Car  Type_U', 'Partner Bin_1', 'Partner Bin_2',
           'Partner Bin_3'],
          dtype='object')




```python
# X_clean = X_clean.drop(['Approve Date', 'Cost of Funds', 'Partner Bin'], axis=1)
XX = XX.drop(['Approve Date', 'Cost of Funds', 'Partner Bin_1', 'Partner Bin_2',
       'Partner Bin_3'], axis=1)
```


```python
feature_names = ['Tier', 'FICO', 'Term', 'Amount',
       'Competition rate', 'Rate', 'Car  Type']
```


```python
from lance import MLClassifier as cl
```


```python
clf = cl.MLClassifiers(XX.drop('Outcome', axis=1), XX['Outcome'])
```


```python
use = clf.available_classifiers[1:3]
trials = 20
```


```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
```


```python
scaler = MinMaxScaler()
clf.run_classifier(n_trials=trials, use_methods=use, scaler=scaler)
```

    Preparing train and test sets...



    HBox(children=(IntProgress(value=0, max=20), HTML(value='')))


    
    Loading models...
    Logistic Regression (L1)



    HBox(children=(IntProgress(value=0, max=400), HTML(value='')))


    
    Logistic Regression (L2)



    HBox(children=(IntProgress(value=0, max=400), HTML(value='')))


    



```python
clf.summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Method</th>
      <th>Test_Accuracy</th>
      <th>Parameter</th>
      <th>Best_Parameter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression (L1)</td>
      <td>0.829377</td>
      <td>C</td>
      <td>29.763514</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression (L2)</td>
      <td>0.829507</td>
      <td>C</td>
      <td>0.127427</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
clf.summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Method</th>
      <th>Test_Accuracy</th>
      <th>Parameter</th>
      <th>Best_Parameter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression (L1)</td>
      <td>0.829377</td>
      <td>C</td>
      <td>29.763514</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression (L2)</td>
      <td>0.829507</td>
      <td>C</td>
      <td>0.127427</td>
    </tr>
  </tbody>
</table>
</div>




```python
clf.plot_accuracy();
```


![png](ML_Lab1_Nomis_files/ML_Lab1_Nomis_30_0.png)



```python
model = clf.methods['Logistic Regression (L2)'].best_model
model_scaler = clf.methods['Logistic Regression (L2)'].scaler
```


```python
X_train, X_val, y_train, y_val = train_test_split(XX.drop('Outcome', axis=1), XX['Outcome'], random_state=42)
```


```python
y_pred = model.predict(model_scaler.transform(X_val))
```


```python
from sklearn.metrics import confusion_matrix
```


```python
confmat = confusion_matrix(y_val, y_pred)
```


```python
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat_U, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat_U.shape[0]):
    for j in range(confmat_U.shape[1]):
        ax.text(x=j, y=i, s=confmat_U[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred))
```


![png](ML_Lab1_Nomis_files/ML_Lab1_Nomis_36_0.png)


                  precision    recall  f1-score   support
    
               0       0.86      0.94      0.90     40653
               1       0.67      0.43      0.53     11369
    
        accuracy                           0.83     52022
       macro avg       0.76      0.69      0.71     52022
    weighted avg       0.82      0.83      0.82     52022
    



```python

```


```python
import ipywidgets as widgets
```


```python
label = widgets.FloatSlider(
    value=0.5,
    min=0,
    max=1,
    step=0.1,
    description='Threshold',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True
)
```


```python
def evaluate_model(label):
    probs = model.predict_proba(model_scaler.transform(X_val))[:,0]
    y3 = (probs < label).astype(int)
    confmat_test = confusion_matrix(y_val, y3)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat_test, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat_test.shape[0]):
        for j in range(confmat_test.shape[1]):
            ax.text(x=j, y=i, s=confmat_test[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()

# from sklearn.metrics import classification_testeport
    print(classification_report(y_val, y3))
```


```python
widgets.interact(evaluate_model, label=label)
```


    interactive(children=(FloatSlider(value=0.5, continuous_update=False, description='Threshold', max=1.0), Outpu…





    <function __main__.evaluate_model(label)>




```python
model.coef_
```




    array([[-0.0310789 , -1.09048689,  1.19532601, -8.23787281,  4.73207767,
             1.22162583, -6.45782547, -0.45133101, -0.618381  ,  1.59102092]])




```python
np.abs(model.coef_[0])
```




    array([0.12325559, 1.09219014, 1.32535349, 8.58144068, 5.70210224,
           1.28773254, 7.02717299, 0.02714433, 0.44262709, 1.96191835])




```python
np.argmax(np.abs(model.coef_[0]))
```




    3




```python
['Amount', 'Previous Rate', 'Competition Rate', 'Rate']
```


```python
XX.drop('Outcome', axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tier</th>
      <th>FICO</th>
      <th>Term</th>
      <th>Amount</th>
      <th>Previous Rate</th>
      <th>Competition rate</th>
      <th>Rate</th>
      <th>Car  Type_N</th>
      <th>Car  Type_R</th>
      <th>Car  Type_U</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>743</td>
      <td>36</td>
      <td>19100.00</td>
      <td>0.00</td>
      <td>4.95</td>
      <td>4.85</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>752</td>
      <td>60</td>
      <td>36460.08</td>
      <td>0.00</td>
      <td>5.65</td>
      <td>5.49</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>778</td>
      <td>48</td>
      <td>12000.00</td>
      <td>0.00</td>
      <td>5.85</td>
      <td>5.85</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>724</td>
      <td>60</td>
      <td>19290.00</td>
      <td>0.00</td>
      <td>5.65</td>
      <td>5.39</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>700</td>
      <td>72</td>
      <td>24461.12</td>
      <td>0.00</td>
      <td>6.25</td>
      <td>6.99</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>756</td>
      <td>66</td>
      <td>12251.00</td>
      <td>0.00</td>
      <td>6.45</td>
      <td>6.49</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>658</td>
      <td>60</td>
      <td>18888.75</td>
      <td>0.00</td>
      <td>5.85</td>
      <td>7.99</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>766</td>
      <td>72</td>
      <td>10555.49</td>
      <td>0.00</td>
      <td>6.45</td>
      <td>6.69</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>779</td>
      <td>72</td>
      <td>20011.59</td>
      <td>0.00</td>
      <td>6.25</td>
      <td>6.59</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>701</td>
      <td>60</td>
      <td>16510.00</td>
      <td>0.00</td>
      <td>5.85</td>
      <td>6.19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3</td>
      <td>758</td>
      <td>60</td>
      <td>44021.44</td>
      <td>0.00</td>
      <td>5.65</td>
      <td>5.49</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>661</td>
      <td>48</td>
      <td>9738.55</td>
      <td>0.00</td>
      <td>5.85</td>
      <td>7.99</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>705</td>
      <td>60</td>
      <td>13024.26</td>
      <td>0.00</td>
      <td>5.85</td>
      <td>6.19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3</td>
      <td>779</td>
      <td>60</td>
      <td>10000.00</td>
      <td>0.00</td>
      <td>5.85</td>
      <td>5.85</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>653</td>
      <td>60</td>
      <td>39000.00</td>
      <td>0.00</td>
      <td>5.65</td>
      <td>7.99</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2</td>
      <td>701</td>
      <td>72</td>
      <td>28506.10</td>
      <td>0.00</td>
      <td>6.25</td>
      <td>6.99</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3</td>
      <td>791</td>
      <td>72</td>
      <td>18861.30</td>
      <td>0.00</td>
      <td>6.45</td>
      <td>6.69</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3</td>
      <td>754</td>
      <td>60</td>
      <td>33206.25</td>
      <td>0.00</td>
      <td>5.65</td>
      <td>5.49</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2</td>
      <td>700</td>
      <td>60</td>
      <td>16500.00</td>
      <td>0.00</td>
      <td>5.85</td>
      <td>6.19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2</td>
      <td>707</td>
      <td>60</td>
      <td>24946.69</td>
      <td>0.00</td>
      <td>5.65</td>
      <td>5.79</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>679</td>
      <td>72</td>
      <td>39000.00</td>
      <td>0.00</td>
      <td>6.25</td>
      <td>7.49</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3</td>
      <td>777</td>
      <td>60</td>
      <td>16986.90</td>
      <td>0.00</td>
      <td>5.85</td>
      <td>5.85</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3</td>
      <td>765</td>
      <td>60</td>
      <td>23337.43</td>
      <td>0.00</td>
      <td>5.85</td>
      <td>5.85</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>698</td>
      <td>72</td>
      <td>46862.36</td>
      <td>0.00</td>
      <td>6.45</td>
      <td>7.69</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3</td>
      <td>739</td>
      <td>36</td>
      <td>22597.47</td>
      <td>0.00</td>
      <td>4.95</td>
      <td>4.85</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>675</td>
      <td>60</td>
      <td>18000.00</td>
      <td>0.00</td>
      <td>5.85</td>
      <td>7.29</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3</td>
      <td>733</td>
      <td>72</td>
      <td>36889.50</td>
      <td>0.00</td>
      <td>6.25</td>
      <td>6.59</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>3</td>
      <td>741</td>
      <td>60</td>
      <td>17372.78</td>
      <td>0.00</td>
      <td>5.65</td>
      <td>5.59</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3</td>
      <td>737</td>
      <td>60</td>
      <td>17042.69</td>
      <td>0.00</td>
      <td>5.65</td>
      <td>5.49</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>3</td>
      <td>778</td>
      <td>60</td>
      <td>17201.32</td>
      <td>0.00</td>
      <td>5.85</td>
      <td>5.85</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>208055</th>
      <td>1</td>
      <td>683</td>
      <td>60</td>
      <td>30000.00</td>
      <td>0.00</td>
      <td>4.45</td>
      <td>8.25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208056</th>
      <td>3</td>
      <td>771</td>
      <td>60</td>
      <td>20600.00</td>
      <td>6.40</td>
      <td>5.55</td>
      <td>4.93</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208057</th>
      <td>1</td>
      <td>696</td>
      <td>60</td>
      <td>20000.00</td>
      <td>0.00</td>
      <td>4.85</td>
      <td>8.49</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>208058</th>
      <td>2</td>
      <td>707</td>
      <td>60</td>
      <td>34999.99</td>
      <td>0.00</td>
      <td>4.45</td>
      <td>6.39</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208059</th>
      <td>2</td>
      <td>737</td>
      <td>72</td>
      <td>20000.00</td>
      <td>0.00</td>
      <td>5.15</td>
      <td>6.69</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208060</th>
      <td>1</td>
      <td>696</td>
      <td>60</td>
      <td>17000.00</td>
      <td>0.00</td>
      <td>4.45</td>
      <td>8.25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208061</th>
      <td>3</td>
      <td>797</td>
      <td>72</td>
      <td>35000.00</td>
      <td>0.00</td>
      <td>5.69</td>
      <td>5.69</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>208062</th>
      <td>3</td>
      <td>753</td>
      <td>48</td>
      <td>14964.00</td>
      <td>5.39</td>
      <td>5.55</td>
      <td>4.75</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208063</th>
      <td>3</td>
      <td>806</td>
      <td>60</td>
      <td>45000.00</td>
      <td>0.00</td>
      <td>4.45</td>
      <td>4.45</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208064</th>
      <td>3</td>
      <td>771</td>
      <td>60</td>
      <td>30074.00</td>
      <td>0.00</td>
      <td>4.45</td>
      <td>4.05</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208065</th>
      <td>1</td>
      <td>717</td>
      <td>72</td>
      <td>24999.99</td>
      <td>0.00</td>
      <td>5.15</td>
      <td>6.79</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208066</th>
      <td>3</td>
      <td>750</td>
      <td>60</td>
      <td>17799.00</td>
      <td>8.99</td>
      <td>5.55</td>
      <td>5.55</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208067</th>
      <td>1</td>
      <td>695</td>
      <td>60</td>
      <td>36000.00</td>
      <td>0.00</td>
      <td>4.45</td>
      <td>8.25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208068</th>
      <td>1</td>
      <td>685</td>
      <td>60</td>
      <td>25000.00</td>
      <td>0.00</td>
      <td>4.45</td>
      <td>8.25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208069</th>
      <td>1</td>
      <td>693</td>
      <td>36</td>
      <td>9999.99</td>
      <td>0.00</td>
      <td>4.35</td>
      <td>8.25</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>208070</th>
      <td>0</td>
      <td>640</td>
      <td>72</td>
      <td>28000.00</td>
      <td>0.00</td>
      <td>5.15</td>
      <td>11.15</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208071</th>
      <td>3</td>
      <td>767</td>
      <td>72</td>
      <td>26729.00</td>
      <td>7.90</td>
      <td>6.35</td>
      <td>6.35</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208072</th>
      <td>3</td>
      <td>749</td>
      <td>48</td>
      <td>9999.99</td>
      <td>0.00</td>
      <td>4.85</td>
      <td>4.85</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>208073</th>
      <td>3</td>
      <td>751</td>
      <td>48</td>
      <td>34428.00</td>
      <td>5.95</td>
      <td>5.55</td>
      <td>4.75</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208074</th>
      <td>3</td>
      <td>777</td>
      <td>48</td>
      <td>35000.00</td>
      <td>0.00</td>
      <td>4.45</td>
      <td>4.45</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208075</th>
      <td>3</td>
      <td>790</td>
      <td>60</td>
      <td>25000.00</td>
      <td>0.00</td>
      <td>4.45</td>
      <td>4.45</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208076</th>
      <td>0</td>
      <td>663</td>
      <td>60</td>
      <td>31800.00</td>
      <td>0.00</td>
      <td>4.45</td>
      <td>10.85</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208077</th>
      <td>3</td>
      <td>742</td>
      <td>60</td>
      <td>33000.00</td>
      <td>0.00</td>
      <td>4.45</td>
      <td>4.45</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208078</th>
      <td>1</td>
      <td>759</td>
      <td>60</td>
      <td>44999.99</td>
      <td>0.00</td>
      <td>4.45</td>
      <td>4.45</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208079</th>
      <td>2</td>
      <td>709</td>
      <td>48</td>
      <td>34999.99</td>
      <td>0.00</td>
      <td>4.45</td>
      <td>6.39</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208080</th>
      <td>1</td>
      <td>685</td>
      <td>60</td>
      <td>20022.00</td>
      <td>8.00</td>
      <td>5.55</td>
      <td>6.53</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208081</th>
      <td>1</td>
      <td>791</td>
      <td>60</td>
      <td>36500.00</td>
      <td>0.00</td>
      <td>4.45</td>
      <td>4.45</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208082</th>
      <td>1</td>
      <td>699</td>
      <td>36</td>
      <td>19999.99</td>
      <td>0.00</td>
      <td>4.35</td>
      <td>8.25</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>208083</th>
      <td>2</td>
      <td>708</td>
      <td>60</td>
      <td>29999.99</td>
      <td>0.00</td>
      <td>4.85</td>
      <td>6.59</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>208084</th>
      <td>3</td>
      <td>780</td>
      <td>60</td>
      <td>34000.00</td>
      <td>0.00</td>
      <td>4.85</td>
      <td>4.85</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>208085 rows × 10 columns</p>
</div>




```python
pwd
```




    '/Users/bengielyndanao/Downloads/Nomis Lab'




```python

```
