---
layout: post
title: Pitfall in Lambda
date: 2018-03-10 17:25:03
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
img: pitfall.jpg # Add image post (optional)
tags: [Python, Lambda, Closure, Partial] # add tag
---

嗨，大家好，這篇又再談談我遇到的另外一個問題：在建置機器學習模型中需要測試多個客製化的量測函式 (_實際的問題我留在下一篇再來講_)，本篇以簡單的例子呈現原始問題的所在：


首先，假設實驗裡面需要測量二維向量的`weighted inner product`，定義一組函式如下，並賦予三組權重`W`來測試一下：

``` python
def w_metric(X, Y, W):
    return sum(x*y*w for x, y, w in zip(X, Y, W))

print(w_metric([1, 2], [4, 5], [1, 1]))
print(w_metric([1, 2], [4, 5], [1, 2]))
print(w_metric([1, 2], [4, 5], [2, 1]))

# >> 14
# >> 24
# >> 18
```

<br>
## Using Lambda
接著，我把三組帶有不同`W`的量測函式物件打包起來，結果輸出不如預期；三組`W`確實在list comprehension裡面被展開，哪裡出錯了？

``` python
def make_w_metric_list(W):
    return [lambda X, Y: w_metric(X, Y, w) for w in W]

metric_list = make_w_metric_list([[1, 1], [1, 2], [2, 1]])

for metric in metric_list:
    print(metric([1, 2], [4, 5]))

# what I expect
# >> 14
# >> 24
# >> 18

# what actually output
# >> 18
# >> 18
# >> 18
```

<br>
## Late Binding - What a Gotcha

吼吼抓到了，透過下面這段程式碼來理解，會發現變數`b`和函式`simple_line`之間的關係不像在C/C++的模式，但程式碼可以正常運作，[Python的解析變數是**當需要用到的時候**][1]，透過[LEGB][2] (local->enclosing->global->built-in) 規則去作name binding。

``` python
def simple_line(x, a):
    return a*x+b

b = 5
print(simple_line(1, 2))
# >> 7

b = 3
print(simple_line(1, 2))
# >> 5
```

相同的規則，回頭去看看原本程式碼，三組函式物件拿到的`w`其實指向`W`最後一個元素`[2, 1]`，透過[live object inspection][3]來驗證一下，`__code__`有編譯後的bytecode資訊，找到`w`在這函式物件裡面是free variables，被closure閉包住了。

``` python
def make_w_metric_list(W):
    return [lambda X, Y: w_metric(X, Y, w) for w in W]

metric_list = make_w_metric_list([[1, 1], [1, 2], [2, 1]])

for metric in metric_list:
    print(metric([1, 2], [4, 5]))
    print(metric.__code__.co_freevars)
    print(metric.__closure__[0].cell_contents)

# >> 18.0
# >> ('w',)
# >> [2.0, 1.0]
# >> 18.0
# >> ('w',)
# >> [2.0, 1.0]
# >> 18.0
# >> ('w',)
# >> [2.0, 1.0]
```

為了幫助理解lambda的行為，原本的函式定義可以等效以較明確的closure方式來呈現，以同樣方式來驗證看看，確實餵進去三組`w_metric`的`w`都指向for loop後的最後一個`w = [2, 1]`。 好，有方法可以解決這種late binding嗎？
``` python
def make_w_metric_list(W):
    func_list = []
    for w in W:
        def _w_metric(X, Y):
            return w_metric(X, Y, w)
        func_list.append(_w_metric)
    return func_list

metric_list = make_w_metric_list([[1, 1], [1, 2], [2, 1]])

for metric in metric_list:
    print(metric([1, 2], [4, 5]))
    print(metric.__code__.co_freevars)
    print(metric.__closure__[0].cell_contents)

# >> 18.0
# >> ('w',)
# >> [2.0, 1.0]
# >> 18.0
# >> ('w',)
# >> [2.0, 1.0]
# >> 18.0
# >> ('w',)
# >> [2.0, 1.0]
```

<br>
## Solution

### **Early Binding - Default Argument**
那就early binding吧，因為Python的機制是[function default argument][4]實際上在definition time就被決定的，所以function的每一次呼叫都是使用同樣的default argument value。
``` python
def make_w_metric_list(W):
    return [lambda X, Y, W=w: w_metric(X, Y, W) for w in W]

metric_list = make_w_metric_list([[1, 1], [1, 2], [2, 1]])

for metric in metric_list:
    print(metric([1, 2], [4, 5]))

# >> 14.0
# >> 24.0
# >> 18.0
```

### **Functool.Partial**
另外，我認為這個方式可讀性比較高，[functool.partial][5]可以把callable的物件重新包裝，並且可以預先設定default argument的值固定住。

``` python
def make_w_metric_list(W):
    return [partial(w_metric, W=w) for w in W]

metric_list = make_w_metric_list([[1, 1], [1, 2], [2, 1]])

for metric in metric_list:
    print(metric([1, 2], [4, 5]))

# >> 14.0
# >> 24.0
# >> 18.0
```

<br>
## Reference
[Python-Execution-Model][1]  
[Python-Scope-of-Variables][2]  
[Python-Live-Object-Inspection][3]  
[Python-Function-Definition][4]  
[Python-Functool-Partial][5]  

[1]: https://docs.python.org/2/reference/executionmodel.html
[2]: https://www.datacamp.com/community/tutorials/scope-of-variables-python
[3]: https://docs.python.org/3/library/inspect.html
[4]: https://docs.python.org/3/reference/compound_stmts.html#function-definitions
[5]: https://docs.python.org/3/library/functools.html#functools.partial