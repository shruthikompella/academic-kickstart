# <span style="color:#d5670f">KNN on IRIS Dataset</span>


```python
# Load CSV
import os
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn import preprocessing
filename = 'file:///C:/Users/shrut/Downloads/iris.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris_dataset = read_csv(filename, names=names)


```

### Let's have a look at the dataset


```python
iris_dataset
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal-length</th>
      <th>sepal-width</th>
      <th>petal-length</th>
      <th>petal-width</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>145</td>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <td>146</td>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <td>147</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <td>148</td>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <td>149</td>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>


### It is very important to well shuffle the dataset to avoid any element of bias/patterns


```python
iris_dataset = iris_dataset.sample(frac=1).reset_index(drop=True)
iris_dataset
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal-length</th>
      <th>sepal-width</th>
      <th>petal-length</th>
      <th>petal-width</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5.5</td>
      <td>2.5</td>
      <td>4.0</td>
      <td>1.3</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <td>1</td>
      <td>5.1</td>
      <td>3.7</td>
      <td>1.5</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <td>2</td>
      <td>6.1</td>
      <td>2.8</td>
      <td>4.0</td>
      <td>1.3</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <td>3</td>
      <td>5.9</td>
      <td>3.0</td>
      <td>4.2</td>
      <td>1.5</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4.8</td>
      <td>3.4</td>
      <td>1.9</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>145</td>
      <td>6.4</td>
      <td>3.2</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <td>146</td>
      <td>5.8</td>
      <td>2.7</td>
      <td>5.1</td>
      <td>1.9</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <td>147</td>
      <td>6.3</td>
      <td>3.4</td>
      <td>5.6</td>
      <td>2.4</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <td>148</td>
      <td>6.6</td>
      <td>2.9</td>
      <td>4.6</td>
      <td>1.3</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <td>149</td>
      <td>6.1</td>
      <td>3.0</td>
      <td>4.6</td>
      <td>1.4</td>
      <td>Iris-versicolor</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
### Adding the "target column" for our convenience to predict the values and have a look at the data


```python
target_dictionary ={'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2} 
  
iris_dataset['target'] = iris_dataset['class'].map(target_dictionary) 
iris_dataset
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal-length</th>
      <th>sepal-width</th>
      <th>petal-length</th>
      <th>petal-width</th>
      <th>class</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5.5</td>
      <td>2.5</td>
      <td>4.0</td>
      <td>1.3</td>
      <td>Iris-versicolor</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>5.1</td>
      <td>3.7</td>
      <td>1.5</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>6.1</td>
      <td>2.8</td>
      <td>4.0</td>
      <td>1.3</td>
      <td>Iris-versicolor</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>5.9</td>
      <td>3.0</td>
      <td>4.2</td>
      <td>1.5</td>
      <td>Iris-versicolor</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4.8</td>
      <td>3.4</td>
      <td>1.9</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>145</td>
      <td>6.4</td>
      <td>3.2</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>Iris-versicolor</td>
      <td>1</td>
    </tr>
    <tr>
      <td>146</td>
      <td>5.8</td>
      <td>2.7</td>
      <td>5.1</td>
      <td>1.9</td>
      <td>Iris-virginica</td>
      <td>2</td>
    </tr>
    <tr>
      <td>147</td>
      <td>6.3</td>
      <td>3.4</td>
      <td>5.6</td>
      <td>2.4</td>
      <td>Iris-virginica</td>
      <td>2</td>
    </tr>
    <tr>
      <td>148</td>
      <td>6.6</td>
      <td>2.9</td>
      <td>4.6</td>
      <td>1.3</td>
      <td>Iris-versicolor</td>
      <td>1</td>
    </tr>
    <tr>
      <td>149</td>
      <td>6.1</td>
      <td>3.0</td>
      <td>4.6</td>
      <td>1.4</td>
      <td>Iris-versicolor</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 6 columns</p>
### Dataframe to Ndarray
> Creating X(attributes) and y(target) and then converting to ndarray for easy calculations



```python
X = iris_dataset[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']].to_numpy()
X[0:5] #printing first 5 rows

```




    array([[5.5, 2.5, 4. , 1.3],
           [5.1, 3.7, 1.5, 0.4],
           [6.1, 2.8, 4. , 1.3],
           [5.9, 3. , 4.2, 1.5],
           [4.8, 3.4, 1.9, 0.2]])




```python
y = iris_dataset['target'].to_numpy()
y #printing y

```




    array([1, 0, 1, 1, 0, 1, 0, 1, 2, 2, 0, 0, 0, 2, 2, 2, 1, 0, 2, 2, 1, 1,
           2, 2, 1, 1, 1, 2, 0, 2, 1, 0, 2, 2, 2, 2, 0, 0, 1, 0, 0, 0, 2, 2,
           2, 1, 0, 1, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 2, 1, 0, 2, 0, 0, 0, 2,
           0, 1, 0, 2, 2, 0, 2, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 1, 1, 0, 1,
           1, 1, 2, 1, 2, 1, 2, 0, 0, 0, 1, 0, 2, 0, 1, 1, 2, 1, 2, 1, 1, 0,
           1, 1, 1, 0, 0, 2, 1, 0, 1, 0, 2, 1, 1, 2, 1, 0, 1, 2, 0, 0, 0, 2,
           2, 2, 0, 1, 1, 1, 0, 2, 0, 2, 2, 2, 0, 1, 2, 2, 1, 1], dtype=int64)



### Splitting the data into Development and test sets


```python
from sklearn.model_selection import train_test_split
X_dev,X_test,y_dev,y_test = train_test_split(X, y, test_size = 0.25, random_state=1)
```

#### Have a look at the shape and analyze after splitting required ratios of devlopments and test sets


```python
print(X_dev.shape)
```

    (112, 4)



```python
print(X_dev[0:5])
```

    [[5.4 3.9 1.7 0.4]
     [6.  2.2 4.  1. ]
     [5.8 2.7 3.9 1.2]
     [5.1 3.5 1.4 0.3]
     [4.7 3.2 1.6 0.2]]



```python
print(y_dev.shape)
```

    (112,)



```python
print(y_dev)
```

    [0 1 1 0 0 1 2 1 2 0 1 1 0 0 0 2 1 2 1 1 1 0 0 0 2 0 1 2 0 2 1 2 1 0 0 0 0
     1 1 1 2 1 0 2 2 1 1 0 1 1 2 2 1 2 0 0 1 1 1 1 0 0 2 0 0 1 0 1 0 2 1 1 1 2
     0 1 2 1 2 1 2 2 1 2 2 1 2 2 0 1 0 2 2 2 0 1 0 0 0 0 0 1 1 0 0 0 0 2 1 2 0
     0]



```python
print(X_test.shape)
print(y_test.shape)
```

    (38, 4)
    (38,)


### Let's just plot the development data by considering two attributes and target value


```python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap=ListedColormap(['#d4040b','#f0b207','#0ee3df'])
plt.figure()
plt.scatter(X_dev[:,0], X_dev[:,1],c=y_dev,cmap=cmap,edgecolor='k',s=50)
plt.show()
```


![Scatterplot](https://shruthikompella.netlify.com/post/knn/output_20_0.png)


### Developed a custom KNN classifier and calculated the distances using following distance metrics 
  1. Euclidean      
  2. Normalized Euclidean  
  3. Cosine Similarity


```python
from collections import Counter
from math import sqrt
import math

def euclidean_distance(x,y,distance_metric):
    if distance_metric == "euclidean" or distance_metric == "normalizedeuclidean": 
        return math.sqrt(sum([(x1-x2)**2 for x1,x2 in zip(x,y)]))      
    elif distance_metric == "cosinesimilarity":
        return (sum([x1*x2 for x1,x2 in zip(x,y)])/(sum([i**2 for i in x]) * sum([i**2 for i in y])))
   
    
class KNN:
    
    def __init__(self,k,metric):
        self.k = k
        self.metric = metric
        
    def fit(self,X,y):
        self.X_dev = X
        self.y_dev = y
    
    def predict(self,X):
        return np.array([self.predict_by(x) for x in X])
        
        
    def predict_by(self,x):
        distances = [euclidean_distance(x, x_dev,self.metric) for x_dev in self.X_dev]   
        # sort the distances
        sort_distances = np.argsort(distances)
        # get k nearest neighbours
        k_indices = sort_distances[:self.k]
        k_nearest_neighbours = [self.y_dev[i] for i in k_indices]
        #print(k_nearest_neighbours)
        # find most occuring class
        most_occuring = Counter(k_nearest_neighbours).most_common(1)
        return most_occuring[0][0]
        
        
        
```

### Calculated the accuracy by iterating all of the development data points for<span style="color:#d5670f"> k = [1,3,5,7]</span> using <span style="color:#d5670f">euclidean distance</span>



```python
j=0
accuracy_euclidean = []
for i in [1,3,5,7]:
    classifier = KNN(i,"euclidean")
    classifier.fit(X_dev,y_dev)
    predictions = classifier.predict(X_dev)
    matches = (predictions == y_dev)
    accuracy_euclidean.append(np.sum(matches)/len(y_dev))
euclidean = {'k': [1,3,5,7],
      'Accuracy': accuracy_euclidean }          
euclidean_data = pd.DataFrame(euclidean)  
euclidean_data
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3</td>
      <td>0.973214</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5</td>
      <td>0.964286</td>
    </tr>
    <tr>
      <td>3</td>
      <td>7</td>
      <td>0.955357</td>
    </tr>
  </tbody>
</table>


### Normalizing the development data


```python
dev_min = X_dev.min(axis=0)
dev_min
```




    array([4.4, 2. , 1. , 0.1])




```python
dev_max = X_dev.max(axis=0)
dev_max
```




    array([7.9, 4.4, 6.9, 2.5])




```python
empty_X_dev = []
for i in range(len(X_dev)):
    for j in range(4):
        empty_X_dev.append((X_dev[i][j]-dev_min[j])/(dev_max[j]-dev_min[j]))

normalized_X_dev = np.asarray(empty_X_dev).reshape(len(X_dev),4)
normalized_X_dev[0:5]

```




    array([[0.28571429, 0.79166667, 0.11864407, 0.125     ],
           [0.45714286, 0.08333333, 0.50847458, 0.375     ],
           [0.4       , 0.29166667, 0.49152542, 0.45833333],
           [0.2       , 0.625     , 0.06779661, 0.08333333],
           [0.08571429, 0.5       , 0.10169492, 0.04166667]])



### Calculated the accuracy by iterating all of the development data points for<span style="color:#d5670f"> k = [1,3,5,7]</span> using <span style="color:#d5670f">normalized euclidean distance</span>



```python
j=0
accuracy_normalized = []
for i in [1,3,5,7]:
    classifier = KNN(i,"normalizedeuclidean")
    classifier.fit(normalized_X_dev,y_dev)
    predictions = classifier.predict(normalized_X_dev)
    matches = (predictions == y_dev)    
    accuracy_normalized.append(np.sum(matches)/len(y_dev))
normalized_euclidean = {'k': [1,3,5,7],
       'Accuracy': accuracy_normalized }          
normalized_euclidean_data = pd.DataFrame(normalized_euclidean)  
normalized_euclidean_data  
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3</td>
      <td>0.955357</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5</td>
      <td>0.955357</td>
    </tr>
    <tr>
      <td>3</td>
      <td>7</td>
      <td>0.955357</td>
    </tr>
  </tbody>
</table>




### Calculated the accuracy by iterating all of the development data points for<span style="color:#d5670f"> k = [1,3,5,7]</span> using <span style="color:#d5670f">cosine similarity</span>



```python
j=0
accuracy_cosine = []
for i in [1,3,5,7]:
    classifier = KNN(i,"cosinesimilarity")
    classifier.fit(X_dev,y_dev)
    predictions = classifier.predict(X_dev)
    #print(predictions)
    matches = (predictions == y_dev)
    accuracy_cosine.append(np.sum(matches)/len(y_dev))
cosine = {'k': [1,3,5,7],
       'Accuracy': accuracy_cosine }          
cosine_data = pd.DataFrame(cosine)  
cosine_data
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0.267857</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3</td>
      <td>0.267857</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5</td>
      <td>0.267857</td>
    </tr>
    <tr>
      <td>3</td>
      <td>7</td>
      <td>0.267857</td>
    </tr>
  </tbody>
</table>


### Drawn bar charts for <span style="color:#d5670f">k vs Accuracy</span>


```python
k_hyperparameter=[1,3,5,7]
import numpy as np
import matplotlib.pyplot as plt
data = [accuracy_euclidean,
accuracy_normalized,
accuracy_cosine]
X = np.arange(len(k_hyperparameter))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
ax.set_ylabel('Accuracy')
ax.set_xlabel("k")
ax.set_xticks(X)
ax.set_xticklabels(k_hyperparameter)
```




    [Text(0, 0, '1'), Text(0, 0, '3'), Text(0, 0, '5'), Text(0, 0, '7')]




![Barchart](https://shruthikompella.netlify.com/post/knn/output_34_1.png)


###  <span style="color:#d5670f">Classify the test data and calculate the accuracy</span>
1. When k=1 we estimate the probability based on a single sample: the closest neighbor. This is very sensitive to all sort of distortions like noise, outliers, mislabelling of data, and so on. By using a higher value for k, it tends to be more robust against those distortions.
2. Also the distance is calculated to itself in k=1 for all distance metrics. Hence we are getting 100% accuracy.
3. It may also be a case of overfitting.
4. Using cosine similarity is leading to poor data labelling.So we dont consider it.

#### Considered k=3 and distance metric to be euclidean for test data


```python
classifier = KNN(3,"euclidean")
classifier.fit(X_dev,y_dev)
predictions = classifier.predict(X_test)
# print(predictions)
matches = (predictions == y_test)
accuracy_test = np.sum(matches)/len(y_test)
accuracy_test
```




    0.9736842105263158

