---
title:  "Logical Not Operation in Numpy"
date:   2018-02-08 13:50:23
categories: [Python]
tags: [Python, Numpy]
---


### Problem Definition
建置預測模型過程中，我把所面臨的問題簡化成分類問題。首先，預處理ground truth data，並將它轉成one-hot vector。
以下使用簡單的範例：

``` python
d = {'d1': [10, 20, 30, 40, 50, 60], 'd2': [25, 11, 20, 30, 40, 50]}
df = pd.DataFrame(data=d)
```
	   d1  d2
	0  10  25
	1  20  11
	2  30  20
	3  40  30
	4  50  40
	5  60  50

我想要把兩個欄位的數據作個轉換，根據threshold分為三個類別，大於10的分到positives，小於-10的分到negatives，至於剩下的就被分到neutrals。 前兩者很好處理，透過`pandas.series`邏輯運算子。

``` python
positives = (df['d1'] - df['d2'] >= 10)
negatives = (df['d1'] - df['d2'] <= -10)
```


<br>
### Logical Not in Numpy Bool
對於neutrals來說，可以透過nor運算來得到。問題來了，以下三種方式到底哪個可以得到預期的效果？
#### 使用Bitwise Not
效果等同於`numpy.invert`，也能得到預期效果。
``` python
neutrals  = ~(positives.values | negatives.values)
```
	[False True False False False False]

#### 使用Logical Not
會造成錯誤，畢竟我的目的是元素與元素的對決。
這個運算子是對於容器整體的判斷，而且必須指明條件，是要**任一個元素**或是**全部元素**都符合才行。
``` python
neutrals  = not(positives.values | negatives.values)
```
    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-22-599dcf4c428e> in <module>()
    ----> 1 neutrals  = not(positives.values | negatives.values)
          2 print(neutrals)


    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

#### 使用Numpy Element-wise Not
使用Numpy內建對於容器的真值判斷含式，這個是元素與元素間的比較，符合需求。
``` python
neutrals  = np.logical_not(positives.values | negatives.values)
```
    [False True False False False False]


<br>
### Logical Not in Python Bool
然而，這樣的情況在Python boolean底下是不同的。
```python
d1 = [10, 20, 30, 40, 50, 60]
d2 = [25, 11, 20, 30, 40, 50]
```


```python
positives = [x-y>=10 for x, y in zip(d1, d2)]
negatives = [x-y<=-10 for x, y in zip(d1, d2)]
```


#### 使用Bitwise Not
注意，Python boolean事實上是用int type來表示的，使用`~`會讓python boolean進行`two's complement`，所以以下的輸出是合理的現象。
```python
neutrals = [~(x|y) for x, y in zip(positives, negatives)]
print(neutrals)
```

    [-2, -1, -2, -2, -2, -2]


#### 使用Logical Not
然而`not`關鍵字能真實反映python boolean於**True/False**的互補。
```python
neutrals = [not(x|y) for x, y in zip(positives, negatives)]
print(neutrals)
```

    [False, True, False, False, False, False]


<br>
### Reference
[Python-numpy-bool][Python-numpy-bool]  
[Python-bitwise][Python-bitwise]  
[Python-logical-not][Python-logical-not]  
[Numpy-bool-negation][Numpy-Bool-Negation]  

[Python-numpy-bool]: http://joergdietrich.github.io/python-numpy-bool-types.html
[Python-bitwise]:   https://stackoverflow.com/questions/791328/how-does-the-bitwise-complement-operator-tilde-work
[Python-logical-not]: https://stackoverflow.com/questions/21415661/logic-operator-for-boolean-indexing-in-pandas/21415990
[Numpy-Bool-Negation]: https://stackoverflow.com/questions/13600988/python-tilde-unary-operator-as-negation-numpy-bool-array