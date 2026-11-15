// 1

import pandas as pd

df=pd.read_csv("archive.zip")
df

df.isnull()

df.isnull().sum()

df.describe()

df.size

df.ndim

df.shape

df.info()

df["study_hours_per_week"]=df["study_hours_per_week"].astype(int)
df

df["attendance_rate"]=df["attendance_rate"].astype(int)
df

df["gender"]=df["gender"].replace({"Female":0,"Male":1})
df

// 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data={
    'student_id':[1,2,3,4,5,6,7,8,9,10],
    'math':[78, 85, np.nan, 90, 120, 65, 70, 200, 88, 76],
    'science':[80, 88, 92, np.nan, 85, 60, 75, 95, 89, 77],
    'english':[75, 82, 78, 85, 90, np.nan, 72, 88, 91, 79],
    'attendence':[85, 90, 95, 80, 105, 70, 75, 60, np.nan, 88]
}
df=pd.DataFrame(data)
df


df.isnull().sum()


df.fillna(df.mean(numeric_only=True),inplace=True)
df


df['math']=df['math'].clip(0,100)
df


df.boxplot()
plt.show()


newdf=df[(df['science']>65) & (df['attendence']>75)]
newdf.boxplot()
plt.show()


df['attendence_log']=np.log(df['attendence'])
df

// 3 

import pandas as pd

# Load Iris dataset directly
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

df = pd.read_csv(url, header=None)

# Display dataset
print(df)

# Display columns
print(df.columns)

# Display species
print(df[4])

# Count species
print(df[4].value_counts())

# Statistical details
print(df.describe())

# Species description
print(df[4].describe())

# Petal width description
print(df[3].describe())

# Group by species
print(df.groupby(4).describe())

# Sum of grouped statistics
print(df.groupby(4).describe().sum())

// or 

import pandas as pd
import zipfile

with zipfile.ZipFile("iris.zip") as z:
    df = pd.read_csv(z.open("iris.data"), header=None)

df
#step 2
df.columns
#step 3
df[4]
#step 4
df[4].value_counts()
#step 5
df.describe()
#step 6
df[4].describe()
#step 7
df[3].describe()
#step 8
df.groupby(4).describe()
#step 9
df.groupby(4).describe().sum()

// 4

# Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Load Dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Step 3: Display Dataset
print("Dataset:")
display(df.head())

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)

# Step 4: Rename Column and Add ID
df.rename(columns={'b': 'black'}, inplace=True)
df.insert(0, 'ID', range(1, len(df) + 1))

# Step 5: Separate Independent and Dependent Variables
x = df[['ID', 'crim', 'zn', 'indus', 'chas', 'nox', 'rm',
        'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']]

y = df['medv']

# Step 6: Split Dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# Step 7: Create Linear Regression Model
model = LinearRegression()

# Step 8: Train Model
model.fit(x_train, y_train)

# Step 9: Predict Values
y_pred = model.predict(x_test)

print("Predicted Values:")
print(y_pred)

# Step 10: Calculate Model Score
print("Training Score:", model.score(x_train, y_train))
print("Testing Score:", model.score(x_test, y_test))

# Step 11: Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error:", rmse)


// 5

# eimport pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score

# step 2: read the dataset
df = pd.read_csv("Social_Network_Ads[1].csv")
df

# step 3: change the gender to numeric datatype
df["Gender"]=df["Gender"].map({"Male":1,"Female":0})
df

# step 4: create x and y variables for test and train a model
x = df[["User ID","Gender","Age","EstimatedSalary"]]
y = df["Purchased"]
x

# step 5: split the dataset into testing and training variables 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=29)

# step 6: create a model
model = LogisticRegression()
model.fit(x_train,y_train)

# step 7: find out the predicted values
y_pred = model.predict(x_test)
y_pred

# step 8 : Find out the all values asked in the question
# model.score(x_train,y_train)
# model.score(x,y)

# cm = confusion_matrix(y_test,y_pred)
# cm

# a = accuracy_score(y_test,y_pred)
# a

# e = 1-a

# p = precision_score(y_test,y_pred)
# p

# r = recall_score(y_test, y_pred)
# r

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# TP, FP, TN, FN
tn, fp, fn, tp = cm.ravel()

print("True Positive (TP):", tp)
print("False Positive (FP):", fp)
print("True Negative (TN):", tn)
print("False Negative (FN):", fn)

# Accuracy
a = accuracy_score(y_test, y_pred)
print("Accuracy:", a)

# Error Rate
e = 1 - a
print("Error Rate:", e)

# Precision
p = precision_score(y_test, y_pred)
print("Precision:", p)

# Recall
r = recall_score(y_test, y_pred)
print("Recall:", r)


// 6

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score

#step 2: load the csv file
df=pd.read_csv("Iris.csv")
df

#step 3: Separate input features and target variable
# x -> Independent variables (features)
# y -> Dependent variable (target/output class)
x=df[['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y=df['Species']
x

#step 4: split the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)

#step 5: create the model 
model=GaussianNB()
model.fit(x_train,y_train)

#step 6: predict the output
y_pred=model.predict(x_test)
y_pred

#step 7: calculate the score
model.score(x_train,y_train)

#step 8: find out the score of original dataset
model.score(x,y)

# step 9: find out the confusion matrix
cm=confusion_matrix(y_test,y_pred)
cm

#step 10: find out the values of fn tn fp tp
total = np.sum(cm)

for i in range(len(cm)):
    TP = cm[i, i]
    FP = np.sum(cm[:, i]) - TP
    FN = np.sum(cm[i, :]) - TP
    TN = total - (TP + FP + FN)
    
    print(f"\nClass {i}: TP={TP}, FP={FP}, FN={FN}, TN={TN}")

#step 11:find out the accuracy
a=accuracy_score(y_test,y_pred)
a

# step 12: error rate
e=1-a
e

// 7

import pandas as pd
import numpy as np
import nltk

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# Step 2: Download packages
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 3: Open the text file
file = open('sample.txt', 'r')

# Step 4: Read the file
content = file.read()
print(content)

# Step 5: Sentence Tokenization
sentence = sent_tokenize(content)
print(sentence)

# Step 6: Word Tokenization
Word = word_tokenize(content)
print(Word)

# Step 7: POS Tagging
pos_tags = nltk.pos_tag(Word)
print(pos_tags)

# Custom Tokenizer
from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer(r"\w+")
wor = tokenize.tokenize(content)
print(wor)

# Step 8: POS Tagging
pos_tags = nltk.pos_tag(Word)
print(pos_tags)

# Step 9: Import Stopwords
from nltk.corpus import stopwords

# Step 10: Get English Stopwords
stop_word = set(stopwords.words('english'))
print(stop_word)

# Step 11: Remove Stopwords
filter_word = [word for word in Word if word.lower() not in stop_word]
print(filter_word)

# Step 12: Stemming
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stemm_word = [stemmer.stem(word) for word in filter_word]
print(stemm_word)

# Step 13: Lemmatization
from nltk.stem import WordNetLemmatizer

lemm = WordNetLemmatizer()
lemm_word = [lemm.lemmatize(word) for word in filter_word]
print(lemm_word)

# Step 14: Processed Text
processed_text = " ".join(lemm_word)
print(processed_text)

# Step 15: Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
tf_matrix = vectorizer.fit_transform([processed_text])

print("Terms:", vectorizer.get_feature_names_out())
print(tf_matrix.toarray())

# Step 16: TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform([processed_text])

print(tfidf_matrix.toarray())

# Close file
file.close()

// 8

import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd

#step 2: load the dataset
df=sns.load_dataset('titanic')
df

#step 3:
df.info()

#step 4:
df.describe()

#step 5:
df.columns

# step 6:
sns.countplot(x='survived',data=df)
plt.title('servival count')
plt.show()

# step 7:
sns.countplot(x='sex',hue='survived',data=df)
plt.title("servival by gender")
plt.show()

# step 8:
sns.histplot(df['fare'],bins=30,kde=True)
plt.title("distribution of tickit Fare")
plt.xlabel("Fare")
plt.ylabel("Frequecy")
plt.show()

// 9

import seaborn as sns
import matplotlib.pyplot as plt

# Step_2
df=sns.load_dataset('titanic')
df

# Step_3
sns.boxplot(x='sex',y='age',hue='survived',data=df)
plt.title('Age Distribution by Gender and Survival Status')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()

// 10
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#step 2
df=pd.read_csv("Iris.csv")
df

#step 3
df.info()

#step 4
# using seaborn
for i in df.columns[:-1]:
    plt.figure(figsize=(6,4))
    sns.histplot(df[i],kde=True,bins=10)

#step 5
for col in df.columns[1:-1]:
    plt.figure(figsize=(6,4))
    sns.boxplot(y=df[col])
    plt.show()
