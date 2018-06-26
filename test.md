

```python
import pandas as pd
import numpy as np
```


```python
d = {'d1': [10, 20, 30, 40, 50, 60], 'd2': [25, 11, 20, 30, 40, 50]}
df = pd.DataFrame(data=d)
df
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
      <th>d1</th>
      <th>d2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>40</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




```python
# prepare for the onehot ground truth vector
positives = (df['d1'] - df['d2'] >= 10)
negatives = (df['d1'] - df['d2'] <= -10)
```


```python
print(type(positives))
print(type(positives.values))
print(type(positives[0]))
```

    <class 'pandas.core.series.Series'>
    <class 'numpy.ndarray'>
    <class 'numpy.bool_'>



```python
neutrals  = ~(positives.values | negatives.values)
print(neutrals)
```

    [False  True False False False False]



```python
neutrals  = not(positives.values | negatives.values)
print(neutrals)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-22-599dcf4c428e> in <module>()
    ----> 1 neutrals  = not(positives.values | negatives.values)
          2 print(neutrals)


    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()



```python
neutrals  = np.logical_not(positives.values | negatives.values)
print(neutrals)
```

    [False  True False False False False]



```python
# prepare for the onehot ground truth vector
positives = (df['d1'] - df['d2'] >= 10)
negatives = (df['d1'] - df['d2'] <= -10)
neutrals  = np.logical_not(positives.values | negatives.values)

print(neutrals)


#positives = np.reshape(positives.values, (-1, 1))
#negatives = np.reshape(negatives.values, (-1, 1))
#neutrals = np.reshape(neutrals, (-1, 1))

df = df.assign(Positives=positives)
df = df.assign(Negatives=negatives)
df = df.assign(Neutrals=neutrals)
df
```

    [False  True False False False False]





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
      <th>d1</th>
      <th>d2</th>
      <th>Positives</th>
      <th>Negatives</th>
      <th>Neutrals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>25</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>11</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>20</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>30</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>40</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>50</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
d1 = [10, 20, 30, 40, 50, 60]
d2 = [25, 11, 20, 30, 40, 50]
```


```python
positives = [x-y>=10 for x, y in zip(d1, d2)]
negatives = [x-y<=-10 for x, y in zip(d1, d2)]
```


```python
print(type(positives))
print(type(negatives))
print(positives)
print(negatives)
```

    <class 'list'>
    <class 'list'>
    [False, False, True, True, True, True]
    [True, False, False, False, False, False]



```python
neutrals = [~(x|y) for x, y in zip(positives, negatives)]
print(neutrals)
```

    [-2, -1, -2, -2, -2, -2]



```python
neutrals = [not(x|y) for x, y in zip(positives, negatives)]
print(neutrals)
```

    [False, True, False, False, False, False]

