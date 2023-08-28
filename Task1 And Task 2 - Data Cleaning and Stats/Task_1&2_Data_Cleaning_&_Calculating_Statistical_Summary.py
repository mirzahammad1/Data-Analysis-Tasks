# import the libararies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the datasets
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# check the datasets
print(train.head())

# data type info
print(train.info())

# checking the shape
print(train.shape)

# --------------------------------------------Preprocessing--------------------------------------------

# combine both datasets
train_len = len(train)
df = pd.concat([train,test],axis=0)
df = df.reset_index(drop=True)
print(df.tail())

# finding null values
print(df.isnull().sum())

# drop the column
df = df.drop(columns=['Cabin'],axis=1)
print(df['Age'].mean())

# fill the missing values using mean for numarical columns
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# fill the missing values using mode for categorical columns
print(df['Embarked'].mode()[0])

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# findind many missing values are there in each column
print(df.isnull().sum())

df['Fare'] = np.log(df['Fare']+1)

# --------------------------------------------Label Encoding--------------------------------------------
from sklearn.preprocessing import LabelEncoder
cols = ['Embarked']
le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])

print(df.head())


df.to_csv('dataset/titanic.csv')
cols = ['Sex']
le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])

df = df.drop(columns=['Name','Ticket'])
print(df.head())

# --------------------------------------------Train-Test split--------------------------------------------
train = df.iloc[:train_len, :]
test = df.iloc[train_len:, :]
print(test.head())

# split inputs
X = train.drop(columns=['PassengerId','Survived'])
Y = train['Survived']
print(X.head())

#  --------------------------------------------Model Training--------------------------------------------
from sklearn.model_selection import train_test_split,cross_val_score
# classify column
def classify (model):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)
    model.fit(x_train, y_train)
    print('Accuracy:', model.score(x_test, y_test))
    
    score = cross_val_score(model, X, Y, cv=5)
    print('CV Score:', np.mean(score))

# checking different model to find which has the highestest score
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.tree import DecisionTreeClassifier 
model = DecisionTreeClassifier()
print(classify(model))

from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier()
print(classify(model))


from sklearn.ensemble import ExtraTreesClassifier 
model = ExtraTreesClassifier()
print(classify(model))

from lightgbm import LGBMClassifier
model = LGBMClassifier()
print(classify(model))

#  --------------------------------------------Complete Model Training with full Data--------------------------------------------
model = LGBMClassifier()
model.fit(X,Y)
print(test.head())

X_test = test.drop(columns=['PassengerId','Survived'])
print(X_test.head())

pred = model.predict(X_test)
print(pred)

# ----------------------------------------------------------------------------------------
sub = pd.read_csv('dataset/submission.csv')
print(sub.head())

sub['Survived'] = pred
sub['Survived'] = sub['Survived'].astype('int')
print(sub.head())

sub.to_csv('dataset/submission.csv')