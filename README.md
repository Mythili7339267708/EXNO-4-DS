
<H3>NAME: V MYTHILI</H3>
<H3>REG NO: 212223040123</H3>
<H3>EX NO: 4</H3>

# Feature Scaling and Selection


# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

# Feature Scaling:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
```

```
data = pd.read_csv('income(1) (1) (1).csv',na_values=["?"])
data
```


![image](https://github.com/user-attachments/assets/0d1b73dc-6ca3-4564-80e6-3756f69b5685)



```
data.isnull().sum()
```

![image](https://github.com/user-attachments/assets/7d8ce515-feae-4e3e-b198-9d3cd14d6af5)


```
m = data[data.isnull().any(axis=1)]
m
```

![image](https://github.com/user-attachments/assets/51be0af9-fb19-4fa1-9a64-36768c090642)

```
data2=data.dropna(axis=0)
data2
```


![image](https://github.com/user-attachments/assets/a07a9b96-7310-43e0-85a5-f9b14f0c10dc)

```
sal = data['SalStat']
data2['SalStat'] = data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

![image](https://github.com/user-attachments/assets/b7a9bc35-7371-47a0-9310-2bbfb6e60b93)

```
sal2=data2['SalStat']
sal = data['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![image](https://github.com/user-attachments/assets/8644b554-0fb8-4c9b-b52d-93f3ccdc5e0c)


```
data2
```

![image](https://github.com/user-attachments/assets/88df1e07-14a2-4e3e-b23c-47cd05124c55)


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

![image](https://github.com/user-attachments/assets/f05fc2bd-ff2f-4b6e-a67c-78abba27de79)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/ef58fee0-5e1d-4341-b9fc-c14d691f8063)


```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![image](https://github.com/user-attachments/assets/e53ab964-690d-40e0-8da7-613ee3683b8e)


```
y=new_data['SalStat'].values
print(y)
```

![image](https://github.com/user-attachments/assets/41e7c6dd-05ec-4756-9a48-07d1bc74b9a4)

```
features=list(set(columns_list)-set(['SalStat']))
x=new_data[features].values
print(x)
```

![image](https://github.com/user-attachments/assets/cc104885-55c7-49bf-9ed0-c0973cb5e8e6)

```
# splitting the data int train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
#fitting the values for x and y
KNN_classifier.fit(train_x,train_y)
```

![image](https://github.com/user-attachments/assets/01aea97f-1b1e-4233-b595-99f7d362b5a6)

```
# predicting the test values with values
prediction = KNN_classifier.predict(test_x)
# performance metric ckeck
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```

![image](https://github.com/user-attachments/assets/21b02fc7-04bf-4525-8fbc-ed5bf5e73996)

```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```

![image](https://github.com/user-attachments/assets/c410a188-3cfa-4314-8944-9040639dcebc)


```
print('Misclassified samples: %d' % (test_y != prediction).sum())
```

![image](https://github.com/user-attachments/assets/9daf6906-6b5a-477d-86c9-c23c142f5096)

```
data.shape
```

![image](https://github.com/user-attachments/assets/3dca7cdf-dbdb-4d64-8ebf-211a9f96e375)


# Feature selection:

```
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2
```
```
df=pd.read_csv('titanic_dataset.csv')
```

```
df.columns
```

![image](https://github.com/user-attachments/assets/b53d7a3d-9c0e-4db3-93eb-c80ae2b48020)

```
df.shape
```
![image](https://github.com/user-attachments/assets/7e158c0c-7f4d-48b3-b94c-3b3f99a46440)

```
x = df.drop("Survived",axis=1)
y = df['Survived']
df1=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)
df1.columns
```
![image](https://github.com/user-attachments/assets/d7d3a40d-ebe6-4c41-ac88-c2985d7f1a43)

```
df1['Age'].isnull().sum()
```
![image](https://github.com/user-attachments/assets/5e6c5d3b-7882-4ee3-b18b-543797cd5f9b)

```
df1['Age'].fillna(method='ffill')
```

![image](https://github.com/user-attachments/assets/31c67b0c-1e11-4583-97e9-405169c13e2c)

```
feature=SelectKBest(mutual_info_classif,k=3)
df1.columns
```
![image](https://github.com/user-attachments/assets/fa72097d-75bd-4801-9a4d-91d12f3e6c3d)

```
cols = df1.columns.tolist()
cols[-1],cols[1] = cols[1], cols[-1]
df1 = df1[cols]
df1.columns
```
![image](https://github.com/user-attachments/assets/fee9ce68-9202-4d55-a16d-428cebe344b7)

```
x=df1.iloc[:,0:6]
y=df1.iloc[:,6]
x.columns
```
![image](https://github.com/user-attachments/assets/8f44af11-a0ab-4367-854d-93ca35463f95)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder
data = pd.read_csv('titanic_dataset.csv')
data=data.dropna()
```
```
# seperate the features and target variables
x = data.drop(['Survived','Name','Ticket'],axis=1)
y = data['Survived']
x
```
![image](https://github.com/user-attachments/assets/f87d4cc4-3e7c-4630-8b84-7f89cf54fa8f)

```
data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")
```
```
data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data["Embarked"]=data["Embarked"].cat.codes
```
```
data
```
![image](https://github.com/user-attachments/assets/47a20d92-3b4a-4c79-a98b-b069b6019ef9)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

# Load the Titanic dataset
data = pd.read_csv('titanic_dataset.csv')

# Display the first few rows of the dataset (optional)
print(data.head())

# Step 1: Select features and target variable
# You may need to adjust these columns based on your dataset
x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]  # Features
y = data['Survived']  # Target variable

# Step 2: Handle missing values if necessary (optional but recommended)
# Fill missing values for 'Age' and 'Embarked' or drop rows with NaN values
x['Age'].fillna(x['Age'].median(), inplace=True)  # Filling missing ages with the median
x['Fare'].fillna(x['Fare'].median(), inplace=True)  # Filling missing fares with the median
x.dropna(inplace=True)  # Drop rows with any other NaN values

# Step 3: One-Hot Encoding for categorical variables
x = pd.get_dummies(x, drop_first=True)  # Converts 'Sex' and 'Pclass' to binary columns

# Step 4: Apply SelectKBest
k = 5  # Choose the number of top features you want to select
selector = SelectKBest(score_func=chi2, k=k)
x_new = selector.fit_transform(x, y)

# Step 5: Get the names of the selected features
selected_features = selector.get_feature_names_out(input_features=x.columns)

# Convert to DataFrame for better readability
x_new_df = pd.DataFrame(x_new, columns=selected_features)

print("Selected Features:")
print(x_new_df)

```

![image](https://github.com/user-attachments/assets/3e1e5c7a-39d0-4ac8-a480-9bd8d425e694)

```
selected_feature_indices = selector.get_support(indices=True)

selected_features = x.columns[selected_feature_indices]

#Print the selected feature names

print("Selected Features:")

print(selected_features)
```
![image](https://github.com/user-attachments/assets/fdcc9ab0-db41-47c6-9b9d-dd1c9a45d18c)

```
x.info
```
![image](https://github.com/user-attachments/assets/baf4e8a2-fc73-405c-97ab-5cf4a24cc5fd)

```
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=5)
x_new = selector.fit_transform(x, y)

#Get the selected feature indices

selected_feature_indices = selector.get_support(indices=True)

#Print the selected feature names

selected_features = x.columns[selected_feature_indices]
print("Selected Features:") 
print(selected_features)
```
![image](https://github.com/user-attachments/assets/9478a68b-dab0-459e-85f9-c82e12bf4d2d)

```
from sklearn.feature_selection import SelectKBest, mutual_info_classif

selector = SelectKBest(score_func=mutual_info_classif, k=5)
x_new = selector.fit_transform(x, y)

# Get the selected feature indices

selected_feature_indices = selector.get_support(indices=True)

# Print the selested feature names

selected_features = x.columns[selected_feature_indices]

print("Selected Features:")

print(selected_features)
```
![image](https://github.com/user-attachments/assets/dc44dc60-d57d-4329-bcc9-e5c61bf1f5e6)

```
from sklearn.feature_selection import SelectPercentile, chi2

selector = SelectPercentile(score_func=chi2, percentile=10) #10% of the features
x_new = selector.fit_transform(x, y)
```
```
import pandas as pd 
from sklearn.feature_selection import SelectFromModel 
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

sfm = SelectFromModel(model, threshold='mean')

sfm.fit(x, y)
selected_features = x.columns[sfm.get_support()]

print("Selected Features:")

print(selected_features)
```
![image](https://github.com/user-attachments/assets/5e827d33-e931-4fbc-a935-39ff7d7733b4)

```
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to your data
model.fit(x, y)

#Get the feature importances 
feature_importances = model.feature_importances_
threshold = 0.15 # Adjust the threshold as needed

#Get the selected features 
selected_features = x.columns[feature_importances > threshold]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/07bfc773-bcf6-4fc5-984c-c4ef964a6a1f)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
df=pd.read_csv('titanic_dataset.csv')
```
```
df.columns
```
![image](https://github.com/user-attachments/assets/00202263-c2c9-409f-a1a7-fbbdae7af7e0)

```
df
```
![image](https://github.com/user-attachments/assets/e8d303f4-07fc-4fdc-92cc-9ee051e864cd)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# Load your Titanic dataset
df = pd.read_csv('titanic_dataset.csv')

# Separate features and target
X = df[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# Fill missing values
X['Age'].fillna(X['Age'].median(), inplace=True)  # Fill NaN in 'Age' with the median
X['Fare'].fillna(X['Fare'].median(), inplace=True)  # Fill NaN in 'Fare' with the median
X['Pclass'].fillna(X['Pclass'].mode()[0], inplace=True)  # Fill NaN in 'Pclass' with the mode (most frequent value)

# SelectKBest with f_classif for feature selection
selector = SelectKBest(score_func=f_classif, k=4)
X_new = selector.fit_transform(X, y)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Print the selected features
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

```

![image](https://github.com/user-attachments/assets/7ae6f68b-fdb8-46d7-ab73-450b1856fd46)

# RESULT:
       The Feature scaling and feature selection executed successfully for the given data.
