---
title: "Urbanization in Metro Manila: The Not-So Livable Metropolis"
excerpt: "A story on how urbanization affects the lives of people. This short project examines the challenges that demand significant changes to basic urban management and policies.<br/><img src='/images/dvs/urbanization-titlecard.png'>"
collection: portfolio
---

<h2>Overview</h2>
<p>This was a final project output for our <b>Data Visualization & Storytelling</b> class under Prof. Erika Legara in the M.Sc. Data Science program. We collected data on the Philippine's historical population and GDP to understand the effect of urbanization. For this project, we used Tableau for the visualizations. This was presented to class last April 2019. </p>

<h2>Report</h2>
As the global landscape continues to evolve, one key theme in driving change is urbanization. Socioeconomic aspects, such as the presence of more jobs and better pay, draw majority of the world's population towards urban areas. 

### Loading Preliminaries


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import re
import datetime

from warnings import filterwarnings
filterwarnings('ignore')
```


```python
plt.style.use('https://gist.githubusercontent.com/lpsy/'
              'e81ff2c0decddc9c6dfeb2fcffe37708/raw/lsy_personal.mplstyle')
```

### Machine Learning Packages


```python
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier
from sklearn import metrics, model_selection
from sklearn.model_selection import GridSearchCV, train_test_split
```

### Load Data


```python
df = pd.read_excel('XAUUSD60 2015-2018 Submission.xlsx',
                   index_col='datetime')
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
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>ceiling</th>
      <th>floor</th>
      <th>ma_short</th>
      <th>ma_short2</th>
      <th>ma_mid</th>
      <th>ma_long</th>
      <th>trigger</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-09-10 01:00:00</th>
      <td>1106.40</td>
      <td>1107.52</td>
      <td>1106.06</td>
      <td>1107.06</td>
      <td>775</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1107.470000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-09-10 02:00:00</th>
      <td>1107.07</td>
      <td>1107.91</td>
      <td>1106.93</td>
      <td>1107.81</td>
      <td>414</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1107.583333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-09-10 03:00:00</th>
      <td>1107.81</td>
      <td>1108.32</td>
      <td>1106.28</td>
      <td>1107.18</td>
      <td>1081</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1107.448889</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-09-10 04:00:00</th>
      <td>1107.23</td>
      <td>1107.23</td>
      <td>1103.87</td>
      <td>1106.81</td>
      <td>2739</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1107.235926</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-09-10 05:00:00</th>
      <td>1106.81</td>
      <td>1107.72</td>
      <td>1106.12</td>
      <td>1106.93</td>
      <td>1517</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1107.133951</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Dataset is an hourly time series from Sep 10, 2015 to June 15, 2018 for a total of 16,298 observations/rows.

Columns:

datetime - Year/Month/Day/Hour  
open - opening price within the hour  
high - highest price within the hour  
low - lowest price for within hour  
close - closing price within the hour  
volume - number of trades within the hour  
ceiling - highest price of the day before  
floor - lowest price of the day before  
ma_short - moving average for a certain number of days (Non-Disclosure Agreement)  
ma_short2 - moving average for a certain number of days (Non-Disclosure Agreement)  
ma_mid - moving average for a certain number of days (Non-Disclosure Agreement)  
ma_long - moving average for a certain number of days (Non-Disclosure Agreement)  
trigger - a certain technical indicator (Non-Disclosure Agreement)  

### Exploratory Data Analysis


```python
fig, ax = plt.subplots()

sns.lineplot(x=df.index, y='close', data=df, ax=ax)
```




![png](/images/forex/forex-eda.png)


The whole dataset is on an upward trend but mostly ranging with numerous dips


```python
fig, ax = plt.subplots(1,2, figsize=(16,9))

#OHLC boxplot
df[['open', 'high', 'low', 'close']].boxplot(ax=ax[0])

#boxplot for volume
df[['volume']].boxplot(ax=ax[1]);
```




![png](/images/forex/forex-boxplot.png)


### Methodology


```python
def pcc(y, factor=1.25):
        """
        Calculates the proportional chance criteria for the dataset

        proportional by chance accuracy rate computes the highest possible
        random chance of classifying data without explicit mathematical
        model other than population counts. As a heuristic or rule of
        thumb, a classifier machine learning model is considered highly
        succcesful when the test accuracy exceeds 1.25*Pcc

        PARAMETERS
        ----------
        factor: float
            Applied factor to the Pcc to determine target test accuracy

        RETURNS
        -------
        pcc: float
            The proportional chance criteria
        """
        counts = np.unique(y, return_counts=True)[1]
        num = (counts/counts.sum())**2

        pcc = 100*factor*num.sum()

        return pcc
```

Set-up the features

- 21 hour lag
- moving averages
- NDA feature (trigger)
- highest attainable price in the next 4 hours


```python
def generate_features(df, window_range=5):
    """ Generate features for a stock/index/currency/commodity based on 
    historical price and performance

    PARAMETERS
    ----------
        df : pandas DataFrame
            dataframe with columns "open", "close", "high", "low", "volume"
        window_range: int, optional (default=5)
            lagged time to include in the features

    RETURNS
    -------
        df_new : pandas DataFrame
            data set with new features
    """

    df_new = pd.DataFrame(index=df.index)

    # original features lagged
    for window in range(1, window_range+1):
        df_new[f'open_lag_{window}'] = df['open'].shift(window)
        df_new[f'close_lag_{window}'] = df['close'].shift(window)
        df_new[f'high_lag_{window}'] = df['high'].shift(window)
        df_new[f'low_lag_{window}'] = df['low'].shift(window)
        df_new[f'vol_lag_{window}'] = df['volume'].shift(window)

    # special features
    df_new['trigger'] = df['trigger'].shift(1)
    df_new['ema_lagged'] = df['ma_short'].shift(1)

    # average price
    df_new['MA_5'] = df['close'].rolling(window=5).mean().shift(1)
    df_new['MA_21'] = df['close'].rolling(window=21).mean().shift(1)

    # average price ratio
    df_new['ratio_MA_5_21'] = df_new['MA_5'] / df_new['MA_21']

    # standard deviation of prices
    df_new['std_price_5'] = df['close'].rolling(window=5).std().shift(1)
    df_new['std_price_21'] = df['close'].rolling(window=21).std().shift(1)

    # standard deviation ratio of prices
    df_new['ratio_std_price_5_21'] = (df_new['std_price_5']
                                      / df_new['std_price_21'])

    df_new = df_new.dropna(axis=0)

    # targets, check the highest gold price attained in the next 4 hours
    highs = df['high'].rolling(window=4).max().shift(-4)
    df_new['high_close_diff'] = highs - df['close'].shift(1)

    return df_new


data = generate_features(df, window_range=21)
```

Set the targets.

Target is an increase of 300 pips (\\$3) in the next 4 hours. An additional 30 pips (\\$0.3) is added to account for the difference in buying and selling prices. The difference is the spread for the forex brokerage.


```python
def reco(x):
    if x.high_close_diff > 3.3:
        return 1
    else:
        return 0
```


```python
data['target'] = data.apply(reco, axis=1)
```


```python
targets = data['target']
X = data.drop(['target', 'high_close_diff'], axis=1)
```

Split the data (Train-Test)


```python
startDATE = datetime.datetime(2015, 9, 11, 1)
testDATE = datetime.datetime(2018, 1, 18, 1)

X_train = X.loc[startDATE:testDATE]
y_train = targets.loc[startDATE:testDATE]

X_test = X.loc[testDATE:]
y_test = targets.loc[testDATE:]

X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((13795, 113), (2420, 113), (13795,), (2420,))



PCC without SMOTE


```python
pcc(y_train, factor=1.25)
```




    67.79956558528475




```python
np.unique(y_train, return_counts=True)
```




    (array([0, 1]), array([8906, 4889]))



Importing SMOTE (Synthetic Minority Oversampling Technique), by the name itself is to oversample for the purpose of balancing the data.


```python
from imblearn import over_sampling
```


```python
sampler = over_sampling.SMOTE()
X_sampled, y_sampled = sampler.fit_resample(X_train, y_train)
```


```python
np.unique(y_sampled, return_counts=True)
```




    (array([0, 1]), array([8906, 8906]))



## Various Classifier Models

### Logistic Regression


```python
log_reg = LogisticRegression(max_iter=1000)

params_log = {'C': [0.001, 0.01, 0.1, 1],
                   'penalty':['l2']}

log_gs = GridSearchCV(log_reg,    
                        param_grid=params_log,
                        scoring='precision', 
                        cv=5, 
                        verbose=10,
                        n_jobs=-1)

log_gs.fit(X_sampled, y_sampled) 

print("Best parameters found: ", log_gs.best_params_)

log_best = log_gs.best_estimator_

predictions_log = log_best.predict(X_test)

print(metrics.classification_report(y_test, predictions_log))

```

    Fitting 5 folds for each of 4 candidates, totalling 20 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:   16.0s
    [Parallel(n_jobs=-1)]: Done   8 out of  20 | elapsed:   26.9s remaining:   40.3s
    [Parallel(n_jobs=-1)]: Done  11 out of  20 | elapsed:   33.2s remaining:   27.2s
    [Parallel(n_jobs=-1)]: Done  14 out of  20 | elapsed:   42.1s remaining:   18.1s
    [Parallel(n_jobs=-1)]: Done  17 out of  20 | elapsed:   43.8s remaining:    7.7s
    [Parallel(n_jobs=-1)]: Done  20 out of  20 | elapsed:   48.3s remaining:    0.0s
    [Parallel(n_jobs=-1)]: Done  20 out of  20 | elapsed:   48.3s finished
    /anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)


    Best parameters found:  {'C': 0.01, 'penalty': 'l2'}
                  precision    recall  f1-score   support
    
               0       0.75      0.71      0.73      1684
               1       0.41      0.46      0.43       737
    
        accuracy                           0.64      2421
       macro avg       0.58      0.59      0.58      2421
    weighted avg       0.65      0.64      0.64      2421
    


### Linear SVM with L2 Regularization


```python
lsvm = LinearSVC(max_iter=10000)

lsvm_param_grid = {'C': [1e-5, 1e-4, 1e-2]}

lsvm_gs = GridSearchCV(lsvm,    
                        param_grid=lsvm_param_grid,
                        scoring='precision_macro', 
                        cv=5, 
                        verbose=1,
                       n_jobs=-1)

lsvm_gs.fit(X_sampled, y_sampled) 

print("Best parameters found: ", lsvm_gs.best_params_)

lsvm_best = lsvm_gs.best_estimator_

predictions_lsvm = lsvm_best.predict(X_test.values)

print(metrics.classification_report(y_test, predictions_lsvm))

```

    Fitting 5 folds for each of 3 candidates, totalling 15 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  15 out of  15 | elapsed:  3.6min finished


    Best parameters found:  {'C': 0.0001}
                  precision    recall  f1-score   support
    
               0       0.74      0.76      0.75      1684
               1       0.42      0.39      0.40       737
    
        accuracy                           0.65      2421
       macro avg       0.58      0.58      0.58      2421
    weighted avg       0.64      0.65      0.64      2421
    


    /anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


### Random Forest


```python
rf = RandomForestClassifier()

rf_param_grid = {'max_depth': range(8,11), 'n_estimators': range(100,201,20)}

rf_gs = GridSearchCV(rf,    
                        param_grid=rf_param_grid,
                        scoring='precision_macro', 
                        cv=5, 
                        verbose=1,
                       n_jobs=-1)

rf_gs.fit(X_sampled, y_sampled) 

print("Best parameters found: ", rf_gs.best_params_)

rf_best = rf_gs.best_estimator_

predictions_rf = rf_best.predict(X_test.values)

print(metrics.classification_report(y_test, predictions_rf))
```

    Fitting 5 folds for each of 18 candidates, totalling 90 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done  90 out of  90 | elapsed:  3.3min finished


    Best parameters found:  {'max_depth': 10, 'n_estimators': 180}
                  precision    recall  f1-score   support
    
               0       0.73      0.79      0.76      1684
               1       0.40      0.32      0.36       736
    
        accuracy                           0.65      2420
       macro avg       0.56      0.56      0.56      2420
    weighted avg       0.63      0.65      0.64      2420
    


### Gradient Boosting


```python
gb = GradientBoostingClassifier()

gb_param_grid = {'max_depth': range(8,11), 'n_estimators': range(100,201,20)}

gb_gs = GridSearchCV(gb,    
                        param_grid=gb_param_grid,
                        scoring='precision_macro', 
                        cv=5, 
                        verbose=1,
                       n_jobs=-1)

gb_gs.fit(X_sampled, y_sampled) 

print("Best parameters found: ", gb_gs.best_params_)

gb_best = gb_gs.best_estimator_

predictions_gb = gb_best.predict(X_test.values)

print(metrics.classification_report(y_test, predictions_gb))
```

    Fitting 5 folds for each of 18 candidates, totalling 90 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed: 11.7min
    [Parallel(n_jobs=-1)]: Done  90 out of  90 | elapsed: 37.6min finished


    Best parameters found:  {'max_depth': 10, 'n_estimators': 160}
                  precision    recall  f1-score   support
    
               0       0.71      0.78      0.75      1684
               1       0.36      0.27      0.31       736
    
        accuracy                           0.63      2420
       macro avg       0.53      0.53      0.53      2420
    weighted avg       0.60      0.63      0.61      2420
    


## Results

### Simulate trades using LSVM

Even if they both have a 65% accuracy, we choose linear svm over random forest as it has a better precision and recall for class 1.


```python
pred_df = df.loc[testDATE:valiDATE]
pred_df['label'] = y_test
pred_df['pred'] = lsvm_best.predict(X_test)
pred_df.head()
```

    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until





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
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>ceiling</th>
      <th>floor</th>
      <th>ceiling2</th>
      <th>floor2</th>
      <th>boxnumber</th>
      <th>...</th>
      <th>B4.1</th>
      <th>B5.1</th>
      <th>B6.1</th>
      <th>trigger</th>
      <th>target100</th>
      <th>target150</th>
      <th>target200</th>
      <th>Unnamed: 46</th>
      <th>label</th>
      <th>pred</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-18 01:00:00</th>
      <td>1326.71</td>
      <td>1328.91</td>
      <td>1326.71</td>
      <td>1328.59</td>
      <td>2199</td>
      <td>1343.94</td>
      <td>1325.97</td>
      <td>1346.94</td>
      <td>1322.97</td>
      <td>6.73</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-01-18</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-18 02:00:00</th>
      <td>1328.57</td>
      <td>1329.59</td>
      <td>1327.45</td>
      <td>1328.50</td>
      <td>6744</td>
      <td>1343.94</td>
      <td>1325.97</td>
      <td>1346.94</td>
      <td>1322.97</td>
      <td>6.73</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-01-18</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-18 03:00:00</th>
      <td>1328.52</td>
      <td>1328.58</td>
      <td>1324.27</td>
      <td>1325.92</td>
      <td>7417</td>
      <td>1343.94</td>
      <td>1325.97</td>
      <td>1346.94</td>
      <td>1322.97</td>
      <td>6.73</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-01-18</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-18 04:00:00</th>
      <td>1325.92</td>
      <td>1327.16</td>
      <td>1325.88</td>
      <td>1326.72</td>
      <td>4356</td>
      <td>1343.94</td>
      <td>1325.97</td>
      <td>1346.94</td>
      <td>1322.97</td>
      <td>6.73</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-01-18</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2018-01-18 05:00:00</th>
      <td>1326.69</td>
      <td>1326.82</td>
      <td>1325.26</td>
      <td>1325.54</td>
      <td>3512</td>
      <td>1343.94</td>
      <td>1325.97</td>
      <td>1346.94</td>
      <td>1322.97</td>
      <td>6.73</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-01-18</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>



Simulation:

equity - this is the starting capital in thousands thus USD100,000  
sl_threshold - is the stoploss which is an order to reduce the losses


```python
equity = 100
sl_threshold = 1
committed = False
buy_price = 0
prediction = 0
hold_time = 0
bet = 0
target_price = 0
stop_loss = 0

trade_count = 0
equity_change = []
trade_results = []
delta_results = []
bet_list = []
target_list = []
buy_list = []
selling_list = []
time = []
sl_list = []

for _, hour in pred_df.iterrows():
    if committed:
        if hold_time < 4:
            if hour.high >= target_price:
                # win, sell current holdings
                pip = (target_price - buy_price - 0.3)*100
                equity += pip*bet
                delta_results.append(pip*bet)
                equity_change.append(equity)
                selling_list.append(target_price)
                trade_results.append(1)
                committed = False
            else:
                # did not win yet, keep check stop_loss
                if hour.low < stop_loss:
                    pip = (stop_loss - buy_price - .3)*100
                    equity += pip*bet
                    delta_results.append(pip*bet)
                    selling_list.append(stop_loss)
                    equity_change.append(equity)
                    trade_results.append(0)
                    committed = False
                else:
                    hold_time += 1                    
        else:
            # lose, sell current holdings at open price
            pip = (np.abs(hour.open - buy_price)-0.3)*100
            equity -= pip*bet
            delta_results.append(-pip*bet)
            equity_change.append(equity)
            selling_list.append(hour.open)
            trade_results.append(0)
            committed = False
    else:
        if hour.pred != 0:
            trade_count += 1
            committed = True
            buy_price = hour.open
            bet = equity/10000
            bet_list.append(bet)
            prediction = hour.pred
            hold_time = 0
            
            target_price = buy_price + 3
            stop_loss = buy_price - sl_threshold
            
            buy_list.append(buy_price)
            target_list.append(target_price)
            time.append(hour.name)
            sl_list.append(stop_loss)
        else:
            committed = False
```


```python
xcheck = pd.DataFrame(dict(buy=buy_list,
                           target=target_list,
                           stop_loss=sl_list,
                           sell=selling_list,
                           bet=bet_list,
                           delta=delta_results,
                          equity=equity_change), index=time)
xcheck
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
      <th>buy</th>
      <th>target</th>
      <th>stop_loss</th>
      <th>sell</th>
      <th>bet</th>
      <th>delta</th>
      <th>equity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-18 04:00:00</th>
      <td>1325.92</td>
      <td>1328.92</td>
      <td>1324.92</td>
      <td>1328.92</td>
      <td>0.010000</td>
      <td>2.700000</td>
      <td>102.700000</td>
    </tr>
    <tr>
      <th>2018-01-18 16:00:00</th>
      <td>1330.09</td>
      <td>1333.09</td>
      <td>1329.09</td>
      <td>1329.09</td>
      <td>0.010270</td>
      <td>-1.335100</td>
      <td>101.364900</td>
    </tr>
    <tr>
      <th>2018-01-18 18:00:00</th>
      <td>1329.71</td>
      <td>1332.71</td>
      <td>1328.71</td>
      <td>1328.71</td>
      <td>0.010136</td>
      <td>-1.317744</td>
      <td>100.047156</td>
    </tr>
    <tr>
      <th>2018-01-19 04:00:00</th>
      <td>1330.67</td>
      <td>1333.67</td>
      <td>1329.67</td>
      <td>1329.67</td>
      <td>0.010005</td>
      <td>-1.300613</td>
      <td>98.746543</td>
    </tr>
    <tr>
      <th>2018-01-19 12:00:00</th>
      <td>1335.61</td>
      <td>1338.61</td>
      <td>1334.61</td>
      <td>1334.61</td>
      <td>0.009875</td>
      <td>-1.283705</td>
      <td>97.462838</td>
    </tr>
    <tr>
      <th>2018-01-19 16:00:00</th>
      <td>1334.18</td>
      <td>1337.18</td>
      <td>1333.18</td>
      <td>1333.18</td>
      <td>0.009746</td>
      <td>-1.267017</td>
      <td>96.195821</td>
    </tr>
    <tr>
      <th>2018-01-22 13:00:00</th>
      <td>1333.63</td>
      <td>1336.63</td>
      <td>1332.63</td>
      <td>1332.63</td>
      <td>0.009620</td>
      <td>-1.250546</td>
      <td>94.945276</td>
    </tr>
    <tr>
      <th>2018-01-23 15:00:00</th>
      <td>1336.51</td>
      <td>1339.51</td>
      <td>1335.51</td>
      <td>1335.51</td>
      <td>0.009495</td>
      <td>-1.234289</td>
      <td>93.710987</td>
    </tr>
    <tr>
      <th>2018-01-23 17:00:00</th>
      <td>1334.16</td>
      <td>1337.16</td>
      <td>1333.16</td>
      <td>1337.16</td>
      <td>0.009371</td>
      <td>2.530197</td>
      <td>96.241184</td>
    </tr>
    <tr>
      <th>2018-01-24 03:00:00</th>
      <td>1340.78</td>
      <td>1343.78</td>
      <td>1339.78</td>
      <td>1342.05</td>
      <td>0.009624</td>
      <td>-0.933539</td>
      <td>95.307644</td>
    </tr>
    <tr>
      <th>2018-01-24 11:00:00</th>
      <td>1347.05</td>
      <td>1350.05</td>
      <td>1346.05</td>
      <td>1350.05</td>
      <td>0.009531</td>
      <td>2.573306</td>
      <td>97.880951</td>
    </tr>
    <tr>
      <th>2018-01-24 13:00:00</th>
      <td>1349.70</td>
      <td>1352.70</td>
      <td>1348.70</td>
      <td>1352.70</td>
      <td>0.009788</td>
      <td>2.642786</td>
      <td>100.523736</td>
    </tr>
    <tr>
      <th>2018-01-24 16:00:00</th>
      <td>1352.21</td>
      <td>1355.21</td>
      <td>1351.21</td>
      <td>1355.21</td>
      <td>0.010052</td>
      <td>2.714141</td>
      <td>103.237877</td>
    </tr>
    <tr>
      <th>2018-01-25 10:00:00</th>
      <td>1361.13</td>
      <td>1364.13</td>
      <td>1360.13</td>
      <td>1360.13</td>
      <td>0.010324</td>
      <td>-1.342092</td>
      <td>101.895785</td>
    </tr>
    <tr>
      <th>2018-01-25 16:00:00</th>
      <td>1360.18</td>
      <td>1363.18</td>
      <td>1359.18</td>
      <td>1359.18</td>
      <td>0.010190</td>
      <td>-1.324645</td>
      <td>100.571140</td>
    </tr>
    <tr>
      <th>2018-01-25 18:00:00</th>
      <td>1357.62</td>
      <td>1360.62</td>
      <td>1356.62</td>
      <td>1360.62</td>
      <td>0.010057</td>
      <td>2.715421</td>
      <td>103.286560</td>
    </tr>
    <tr>
      <th>2018-01-25 22:00:00</th>
      <td>1347.71</td>
      <td>1350.71</td>
      <td>1346.71</td>
      <td>1346.71</td>
      <td>0.010329</td>
      <td>-1.342725</td>
      <td>101.943835</td>
    </tr>
    <tr>
      <th>2018-01-26 03:00:00</th>
      <td>1348.26</td>
      <td>1351.26</td>
      <td>1347.26</td>
      <td>1351.26</td>
      <td>0.010194</td>
      <td>2.752484</td>
      <td>104.696319</td>
    </tr>
    <tr>
      <th>2018-01-26 11:00:00</th>
      <td>1355.08</td>
      <td>1358.08</td>
      <td>1354.08</td>
      <td>1354.08</td>
      <td>0.010470</td>
      <td>-1.361052</td>
      <td>103.335266</td>
    </tr>
    <tr>
      <th>2018-01-26 14:00:00</th>
      <td>1350.46</td>
      <td>1353.46</td>
      <td>1349.46</td>
      <td>1353.46</td>
      <td>0.010334</td>
      <td>2.790052</td>
      <td>106.125319</td>
    </tr>
    <tr>
      <th>2018-01-26 16:00:00</th>
      <td>1351.57</td>
      <td>1354.57</td>
      <td>1350.57</td>
      <td>1350.57</td>
      <td>0.010613</td>
      <td>-1.379629</td>
      <td>104.745689</td>
    </tr>
    <tr>
      <th>2018-01-26 18:00:00</th>
      <td>1352.03</td>
      <td>1355.03</td>
      <td>1351.03</td>
      <td>1351.03</td>
      <td>0.010475</td>
      <td>-1.361694</td>
      <td>103.383996</td>
    </tr>
    <tr>
      <th>2018-01-29 03:00:00</th>
      <td>1350.78</td>
      <td>1353.78</td>
      <td>1349.78</td>
      <td>1349.78</td>
      <td>0.010338</td>
      <td>-1.343992</td>
      <td>102.040004</td>
    </tr>
    <tr>
      <th>2018-01-29 11:00:00</th>
      <td>1346.21</td>
      <td>1349.21</td>
      <td>1345.21</td>
      <td>1345.21</td>
      <td>0.010204</td>
      <td>-1.326520</td>
      <td>100.713484</td>
    </tr>
    <tr>
      <th>2018-01-29 15:00:00</th>
      <td>1343.36</td>
      <td>1346.36</td>
      <td>1342.36</td>
      <td>1342.36</td>
      <td>0.010071</td>
      <td>-1.309275</td>
      <td>99.404208</td>
    </tr>
    <tr>
      <th>2018-01-29 19:00:00</th>
      <td>1340.03</td>
      <td>1343.03</td>
      <td>1339.03</td>
      <td>1343.03</td>
      <td>0.009940</td>
      <td>2.683914</td>
      <td>102.088122</td>
    </tr>
    <tr>
      <th>2018-01-30 12:00:00</th>
      <td>1343.80</td>
      <td>1346.80</td>
      <td>1342.80</td>
      <td>1342.80</td>
      <td>0.010209</td>
      <td>-1.327146</td>
      <td>100.760976</td>
    </tr>
    <tr>
      <th>2018-01-30 16:00:00</th>
      <td>1345.90</td>
      <td>1348.90</td>
      <td>1344.90</td>
      <td>1344.90</td>
      <td>0.010076</td>
      <td>-1.309893</td>
      <td>99.451084</td>
    </tr>
    <tr>
      <th>2018-01-31 04:00:00</th>
      <td>1338.72</td>
      <td>1341.72</td>
      <td>1337.72</td>
      <td>1341.72</td>
      <td>0.009945</td>
      <td>2.685179</td>
      <td>102.136263</td>
    </tr>
    <tr>
      <th>2018-01-31 12:00:00</th>
      <td>1343.21</td>
      <td>1346.21</td>
      <td>1342.21</td>
      <td>1342.21</td>
      <td>0.010214</td>
      <td>-1.327771</td>
      <td>100.808491</td>
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
    </tr>
    <tr>
      <th>2018-06-01 16:00:00</th>
      <td>1292.40</td>
      <td>1295.40</td>
      <td>1291.40</td>
      <td>1295.40</td>
      <td>0.014717</td>
      <td>3.973537</td>
      <td>151.141558</td>
    </tr>
    <tr>
      <th>2018-06-01 18:00:00</th>
      <td>1295.02</td>
      <td>1298.02</td>
      <td>1294.02</td>
      <td>1294.02</td>
      <td>0.015114</td>
      <td>-1.964840</td>
      <td>149.176718</td>
    </tr>
    <tr>
      <th>2018-06-04 11:00:00</th>
      <td>1292.81</td>
      <td>1295.81</td>
      <td>1291.81</td>
      <td>1295.81</td>
      <td>0.014918</td>
      <td>4.027771</td>
      <td>153.204489</td>
    </tr>
    <tr>
      <th>2018-06-04 15:00:00</th>
      <td>1296.57</td>
      <td>1299.57</td>
      <td>1295.57</td>
      <td>1295.57</td>
      <td>0.015320</td>
      <td>-1.991658</td>
      <td>151.212831</td>
    </tr>
    <tr>
      <th>2018-06-04 17:00:00</th>
      <td>1295.28</td>
      <td>1298.28</td>
      <td>1294.28</td>
      <td>1294.28</td>
      <td>0.015121</td>
      <td>-1.965767</td>
      <td>149.247064</td>
    </tr>
    <tr>
      <th>2018-06-05 11:00:00</th>
      <td>1291.77</td>
      <td>1294.77</td>
      <td>1290.77</td>
      <td>1290.77</td>
      <td>0.014925</td>
      <td>-1.940212</td>
      <td>147.306852</td>
    </tr>
    <tr>
      <th>2018-06-05 14:00:00</th>
      <td>1290.65</td>
      <td>1293.65</td>
      <td>1289.65</td>
      <td>1293.65</td>
      <td>0.014731</td>
      <td>3.977285</td>
      <td>151.284137</td>
    </tr>
    <tr>
      <th>2018-06-05 17:00:00</th>
      <td>1292.42</td>
      <td>1295.42</td>
      <td>1291.42</td>
      <td>1295.42</td>
      <td>0.015128</td>
      <td>4.084672</td>
      <td>155.368809</td>
    </tr>
    <tr>
      <th>2018-06-05 20:00:00</th>
      <td>1298.41</td>
      <td>1301.41</td>
      <td>1297.41</td>
      <td>1297.41</td>
      <td>0.015537</td>
      <td>-2.019795</td>
      <td>153.349014</td>
    </tr>
    <tr>
      <th>2018-06-06 11:00:00</th>
      <td>1298.18</td>
      <td>1301.18</td>
      <td>1297.18</td>
      <td>1297.18</td>
      <td>0.015335</td>
      <td>-1.993537</td>
      <td>151.355477</td>
    </tr>
    <tr>
      <th>2018-06-06 13:00:00</th>
      <td>1295.00</td>
      <td>1298.00</td>
      <td>1294.00</td>
      <td>1294.00</td>
      <td>0.015136</td>
      <td>-1.967621</td>
      <td>149.387856</td>
    </tr>
    <tr>
      <th>2018-06-06 16:00:00</th>
      <td>1298.13</td>
      <td>1301.13</td>
      <td>1297.13</td>
      <td>1301.13</td>
      <td>0.014939</td>
      <td>4.033472</td>
      <td>153.421328</td>
    </tr>
    <tr>
      <th>2018-06-07 11:00:00</th>
      <td>1298.53</td>
      <td>1301.53</td>
      <td>1297.53</td>
      <td>1301.53</td>
      <td>0.015342</td>
      <td>4.142376</td>
      <td>157.563704</td>
    </tr>
    <tr>
      <th>2018-06-07 16:00:00</th>
      <td>1301.33</td>
      <td>1304.33</td>
      <td>1300.33</td>
      <td>1300.33</td>
      <td>0.015756</td>
      <td>-2.048328</td>
      <td>155.515376</td>
    </tr>
    <tr>
      <th>2018-06-08 11:00:00</th>
      <td>1298.49</td>
      <td>1301.49</td>
      <td>1297.49</td>
      <td>1297.49</td>
      <td>0.015552</td>
      <td>-2.021700</td>
      <td>153.493676</td>
    </tr>
    <tr>
      <th>2018-06-08 16:00:00</th>
      <td>1298.21</td>
      <td>1301.21</td>
      <td>1297.21</td>
      <td>1297.21</td>
      <td>0.015349</td>
      <td>-1.995418</td>
      <td>151.498258</td>
    </tr>
    <tr>
      <th>2018-06-11 11:00:00</th>
      <td>1296.69</td>
      <td>1299.69</td>
      <td>1295.69</td>
      <td>1295.69</td>
      <td>0.015150</td>
      <td>-1.969477</td>
      <td>149.528781</td>
    </tr>
    <tr>
      <th>2018-06-11 13:00:00</th>
      <td>1295.50</td>
      <td>1298.50</td>
      <td>1294.50</td>
      <td>1298.50</td>
      <td>0.014953</td>
      <td>4.037277</td>
      <td>153.566058</td>
    </tr>
    <tr>
      <th>2018-06-11 16:00:00</th>
      <td>1298.89</td>
      <td>1301.89</td>
      <td>1297.89</td>
      <td>1301.89</td>
      <td>0.015357</td>
      <td>4.146284</td>
      <td>157.712341</td>
    </tr>
    <tr>
      <th>2018-06-12 12:00:00</th>
      <td>1297.71</td>
      <td>1300.71</td>
      <td>1296.71</td>
      <td>1296.71</td>
      <td>0.015771</td>
      <td>-2.050260</td>
      <td>155.662081</td>
    </tr>
    <tr>
      <th>2018-06-12 16:00:00</th>
      <td>1295.62</td>
      <td>1298.62</td>
      <td>1294.62</td>
      <td>1298.62</td>
      <td>0.015566</td>
      <td>4.202876</td>
      <td>159.864957</td>
    </tr>
    <tr>
      <th>2018-06-13 13:00:00</th>
      <td>1294.56</td>
      <td>1297.56</td>
      <td>1293.56</td>
      <td>1297.56</td>
      <td>0.015986</td>
      <td>4.316354</td>
      <td>164.181311</td>
    </tr>
    <tr>
      <th>2018-06-13 22:00:00</th>
      <td>1299.69</td>
      <td>1302.69</td>
      <td>1298.69</td>
      <td>1298.69</td>
      <td>0.016418</td>
      <td>-2.134357</td>
      <td>162.046954</td>
    </tr>
    <tr>
      <th>2018-06-14 08:00:00</th>
      <td>1299.35</td>
      <td>1302.35</td>
      <td>1298.35</td>
      <td>1302.35</td>
      <td>0.016205</td>
      <td>4.375268</td>
      <td>166.422222</td>
    </tr>
    <tr>
      <th>2018-06-14 16:00:00</th>
      <td>1307.28</td>
      <td>1310.28</td>
      <td>1306.28</td>
      <td>1306.28</td>
      <td>0.016642</td>
      <td>-2.163489</td>
      <td>164.258733</td>
    </tr>
    <tr>
      <th>2018-06-14 19:00:00</th>
      <td>1304.39</td>
      <td>1307.39</td>
      <td>1303.39</td>
      <td>1303.39</td>
      <td>0.016426</td>
      <td>-2.135364</td>
      <td>162.123369</td>
    </tr>
    <tr>
      <th>2018-06-15 10:00:00</th>
      <td>1299.55</td>
      <td>1302.55</td>
      <td>1298.55</td>
      <td>1298.55</td>
      <td>0.016212</td>
      <td>-2.107604</td>
      <td>160.015765</td>
    </tr>
    <tr>
      <th>2018-06-15 14:00:00</th>
      <td>1299.58</td>
      <td>1302.58</td>
      <td>1298.58</td>
      <td>1298.58</td>
      <td>0.016002</td>
      <td>-2.080205</td>
      <td>157.935560</td>
    </tr>
    <tr>
      <th>2018-06-15 16:00:00</th>
      <td>1292.32</td>
      <td>1295.32</td>
      <td>1291.32</td>
      <td>1291.32</td>
      <td>0.015794</td>
      <td>-2.053162</td>
      <td>155.882398</td>
    </tr>
    <tr>
      <th>2018-06-15 18:00:00</th>
      <td>1280.50</td>
      <td>1283.50</td>
      <td>1279.50</td>
      <td>1279.50</td>
      <td>0.015588</td>
      <td>-2.026471</td>
      <td>153.855927</td>
    </tr>
  </tbody>
</table>
<p>375 rows × 7 columns</p>
</div>




```python
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(range(1, trade_count+1), equity_change)
ax.set_ylabel('Equity (USD)')
ax.set_xlabel('Trades Taken')
ax.grid(False)
ax.yaxis.grid(True)
ax.axis([0,380, 70, 170])

for i, tr in enumerate(trade_results):
    if tr == 1:
        ax.axvline(x=(i+1), c='g', ls='--', lw=0.5)
    else:
        ax.axvline(x=(i+1), c='r', ls='--', lw=0.5)

# ax.spines['left'].set_position(('data', 0))
```


![png](/images/forex/forex-prediction.png)


Actual Simulation:

- target of 300 pips and a 100 stoploss
- period is from Jan 18, 2018 to Jun 15, 2018 or equivalent to six (6) months
- resulting profit is from USD100,000 to USD154,000 or an increase in capital by 54%


```python

```
The full infographic can be accessed [here](/files/urbanization-deck.pdf).

<h2>Acknowledgements</h2>
<p>This project was completed together with my learning teammates Bea Palomar and Nigel Silva.</p>