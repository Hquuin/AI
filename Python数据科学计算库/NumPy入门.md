# NumPy入门

## 1.开发环境安装与配置

## 2.NumPy数组引出

## 3.NumPy数组创建



### **1.array()函数**

```python
numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
```

![image-20221227161034268](NumPy入门.assets/image-20221227161034268.png)

实例：

```python
import numpy as np

# 一维数组
# str>float>int
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))

# 多维数据
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2)

# ndmin参数
arr3 = np.array([1, 2, 3, 4, 5], ndmin=2)
print(arr3)

# dtype参数
arr4 = np.array([1, 2, 3, 4, 5], dtype='f')
print(arr4)

# 结构化数据类型
student = np.dtype([("name", "S20"), ("age", "i4"), ("marks", "f4")])
arr5 = np.array([("jack", 18, 99.9), ('tom', 19, 98.9)], dtype=student)
print(arr5)



输出结果：
[1 2 3 4 5]
<class 'numpy.ndarray'>
[[1 2 3]
 [4 5 6]]
[[1 2 3 4 5]]
[1. 2. 3. 4. 5.]
[(b'jack', 18, 99.9) (b'tom', 19, 98.9)]
```



### **2.empty(函数)**

````python
numpy.empty(shape, dtype = float, order = 'C')
````



### **3.zeros()函数**

```python
numpy.zeros(shape, dtype = float, order = 'C')
```



### **4.ones()函数**

```python
numpy.zeros(shape, dtype = None, order = 'C')
```



![image-20221230102853059](NumPy入门.assets/image-20221230102853059.png)



实例：

```python
# empty()函数
arr6 = np.empty([3, 2], dtype=int)
print(arr6)

# zeros()函数
arr7 = np.zeros(5)
print(arr7)

# zeros()函数
arr8 = np.zeros([3, 2], dtype=[('x', 'i4'), ('y', '<f4')])
print(arr8)

# ones()函数
arr = np.ones(5)
print(arr)

arr = np.ones([2, 2], dtype=int)
print(arr)

输出结果：
[[ 6917529027641081856  5764616291768666155]
 [ 6917529027641081859 -5764598754299804209]
 [          4497473538      844429428932120]]
[0. 0. 0. 0. 0.]
[[(0, 0.) (0, 0.)]
 [(0, 0.) (0, 0.)]
 [(0, 0.) (0, 0.)]]
[1. 1. 1. 1. 1.]
[[1 1]
 [1 1]]
```



### **5.full()函数**

```python
numpy.zeros(shape, fill_value, dtype = None, order = 'C')
```



实例：

```python
# full()函数
arr = np.full(5, fill_value=1024)
print(arr)

arr = np.full([3, 2], fill_value=1024)
print(arr)

输出结果：
[1024 1024 1024 1024 1024]
[[1024 1024]
 [1024 1024]
 [1024 1024]]
```



### **6.eye()函数**

```python
numpy.eye(N, M=None, k=0, dtype = float, order = 'C')
N:行数量
M:列数量，默认等于行数量，可选
```



实例：

````python
# eye()函数
arr = np.eye(10, dtype=int)
print(arr)

输出结果：
[[1 0 0 0 0 0 0 0 0 0]
 [0 1 0 0 0 0 0 0 0 0]
 [0 0 1 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 0 0]
 [0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0]
 [0 0 0 0 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 1 0]
 [0 0 0 0 0 0 0 0 0 1]]
````



### 7.arange()函数

```python
numpy.arange(start, stop, step, dtype)
```

![image-20221230105909873](NumPy入门.assets/image-20221230105909873.png)



实例：

````python
# arange()函数
arr = np.arange(1, 11, 2)
print(arr)

输出结果：
[1 3 5 7 9]
````



### 8.linspace()函数(等差数列)

````python
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
````

![image-20221230110401685](NumPy入门.assets/image-20221230110401685.png)



实例：

```python
# linspace()函数
arr = np.linspace(10, 20, 5, endpoint=True)
print(arr)

输出结果：
[10.  12.5 15.  17.5 20. ]
```



### 9.logspace()函数(等比数列)

### 

````python
np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
````

![image-20221230110920985](NumPy入门.assets/image-20221230110920985.png)



实例：

```python
# logspace()函数
arr = np.logspace(0, 9, 10, base=2)
print(arr)

输出结果：
[  1.   2.   4.   8.  16.  32.  64. 128. 256. 512.]
```









## 4.NumPy数组查看

## 5.NumPy数据保存和数据类型

​     **数据类型：**

![](NumPy入门.assets/image-20221227153407861.png)



![image-20221227153732996](NumPy入门.assets/image-20221227153732996.png)



**数据类型对象 (dtype)**

数据类型对象（numpy.dtype 类的实例）用来描述与数组对应的内存区域是如何使用，它描述了数据的以下几个方面：

- 数据的类型（整数，浮点数或者 Python 对象）
- 数据的大小（例如， 整数使用多少个字节存储）
- 数据的字节顺序（小端法或大端法）
- 在结构化类型的情况下，字段的名称、每个字段的数据类型和每个字段所取的内存块的部分
- 如果数据类型是子数组，那么它的形状和数据类型是什么。



​	

```python
import numpy as np

# int8, int16, int32, int64 四种数据类型可以使用字符串 'i1', 'i2','i4','i8' 代替
dt = np.dtype('i4')
print(dt)
print(type(dt))

输出结果:
int32
<class 'numpy.dtype[int32]'>
```



## 6.NumPy数组运算

## 7.NumPy索引与切片