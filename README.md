# Machine Learning and the Titanic

Import packages


```python
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
```

Import data


```python
df = pd.read_csv('train_and_test2.csv')
```


```python
print(df.columns.values)
```

    ['Passengerid' 'Age' 'Fare' 'Sex' 'sibsp' 'zero' 'zero.1' 'zero.2'
     'zero.3' 'zero.4' 'zero.5' 'zero.6' 'Parch' 'zero.7' 'zero.8' 'zero.9'
     'zero.10' 'zero.11' 'zero.12' 'zero.13' 'zero.14' 'Pclass' 'zero.15'
     'zero.16' 'Embarked' 'zero.17' 'zero.18' '2urvived']



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
      <th>Passengerid</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Sex</th>
      <th>sibsp</th>
      <th>zero</th>
      <th>zero.1</th>
      <th>zero.2</th>
      <th>zero.3</th>
      <th>zero.4</th>
      <th>...</th>
      <th>zero.12</th>
      <th>zero.13</th>
      <th>zero.14</th>
      <th>Pclass</th>
      <th>zero.15</th>
      <th>zero.16</th>
      <th>Embarked</th>
      <th>zero.17</th>
      <th>zero.18</th>
      <th>2urvived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
df.tail()
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
      <th>Passengerid</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Sex</th>
      <th>sibsp</th>
      <th>zero</th>
      <th>zero.1</th>
      <th>zero.2</th>
      <th>zero.3</th>
      <th>zero.4</th>
      <th>...</th>
      <th>zero.12</th>
      <th>zero.13</th>
      <th>zero.14</th>
      <th>Pclass</th>
      <th>zero.15</th>
      <th>zero.16</th>
      <th>Embarked</th>
      <th>zero.17</th>
      <th>zero.18</th>
      <th>2urvived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1304</td>
      <td>1305</td>
      <td>28.0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1305</td>
      <td>1306</td>
      <td>39.0</td>
      <td>108.9000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1306</td>
      <td>1307</td>
      <td>38.5</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1307</td>
      <td>1308</td>
      <td>28.0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1308</td>
      <td>1309</td>
      <td>28.0</td>
      <td>22.3583</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 28 columns):
    Passengerid    1309 non-null int64
    Age            1309 non-null float64
    Fare           1309 non-null float64
    Sex            1309 non-null int64
    sibsp          1309 non-null int64
    zero           1309 non-null int64
    zero.1         1309 non-null int64
    zero.2         1309 non-null int64
    zero.3         1309 non-null int64
    zero.4         1309 non-null int64
    zero.5         1309 non-null int64
    zero.6         1309 non-null int64
    Parch          1309 non-null int64
    zero.7         1309 non-null int64
    zero.8         1309 non-null int64
    zero.9         1309 non-null int64
    zero.10        1309 non-null int64
    zero.11        1309 non-null int64
    zero.12        1309 non-null int64
    zero.13        1309 non-null int64
    zero.14        1309 non-null int64
    Pclass         1309 non-null int64
    zero.15        1309 non-null int64
    zero.16        1309 non-null int64
    Embarked       1307 non-null float64
    zero.17        1309 non-null int64
    zero.18        1309 non-null int64
    2urvived       1309 non-null int64
    dtypes: float64(3), int64(25)
    memory usage: 286.5 KB



```python

```
