// DAA 

// 1 

n = int(input("Enter number of terms: "))

a = 0
b = 1
steps = 0

print("Fibonacci Series:")

for i in range(n):
    print(a, end=" ")
    c = a + b
    a = b
    b = c
    steps += 1

print("\nStep Count:", steps)

// 2 
// A

import heapq

data = {'A': 5, 'B': 9, 'C': 12, 'D': 13}

heap = [[v, k] for k, v in data.items()]
heapq.heapify(heap)

while len(heap) > 1:
    a = heapq.heappop(heap)
    b = heapq.heappop(heap)
    heapq.heappush(heap, [a[0]+b[0], a[1]+b[1]])

print("Huffman Encoding:", heap[0])

// B

jobs = [('A', 2, 100), ('B', 1, 50),
        ('C', 2, 200), ('D', 1, 30)]

jobs.sort(key=lambda x: x[2], reverse=True)

result = [None] * 2
profit = 0

for job in jobs:
    for j in range(job[1]-1, -1, -1):
        if result[j] == None:
            result[j] = job[0]
            profit += job[2]
            break

print("Selected Jobs:", result)
print("Maximum Profit:", profit)

// 3 
weight = [10, 20, 30]
profit = [60, 100, 120]

capacity = 50
total = 0

ratio = []

for i in range(len(weight)):
    ratio.append((profit[i] / weight[i], i))

ratio.sort(reverse=True)

for r, i in ratio:
    if weight[i] <= capacity:
        total += profit[i]
        capacity -= weight[i]
    else:
        total += r * capacity
        break

print("Maximum Profit:", total)

// 4 

weight = [1, 3, 4, 5]
profit = [1, 4, 5, 7]

capacity = 7
n = len(weight)

dp = [[0 for j in range(capacity + 1)]
      for i in range(n + 1)]

for i in range(1, n + 1):
    for j in range(capacity + 1):

        if weight[i-1] <= j:
            dp[i][j] = max(
                profit[i-1] + dp[i-1][j-weight[i-1]],
                dp[i-1][j]
            )
        else:
            dp[i][j] = dp[i-1][j]

print("Maximum Profit:", dp[n][capacity])

// 5 

n = 8
board = [[0 for j in range(n)] for i in range(n)]

board[0][0] = 1

def safe(row, col):

    for i in range(row):
        if board[i][col] == 1:
            return False

    for i, j in zip(range(row-1, -1, -1),
                    range(col-1, -1, -1)):
        if board[i][j] == 1:
            return False

    for i, j in zip(range(row-1, -1, -1),
                    range(col+1, n)):
        if board[i][j] == 1:
            return False

    return True


def solve(row):

    if row == n:
        return True

    for col in range(n):

        if safe(row, col):
            board[row][col] = 1

            if solve(row + 1):
                return True

            board[row][col] = 0

    return False


if solve(1):
    print("8 Queens Solution:")

    for row in board:
        print(row)
else:
    print("No Solution")




// ML

// 1 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

data = {
    'pickup': [1,2,3,4,5,6,7,8,9,10],
    'drop': [2,3,4,5,6,7,8,9,10,11],
    'fare': [50,60,70,80,90,100,110,120,130,140]
}

df = pd.DataFrame(data)

X = df[['pickup','drop']]
y = df['fare']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

for model in [LinearRegression(), RandomForestRegressor(n_estimators=10)]:
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    print("R2:", r2_score(y_test, p))
    print("RMSE:", mean_squared_error(y_test, p)**0.5)

// 2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([
    [1,2], [2,3], [3,2],
    [8,9], [9,8], [8,8],
    [4,5], [5,4], [5,5]
])

wcss = []

for i in range(1, 6):
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(X)
    wcss.append(km.inertia_)

plt.plot(range(1,6), wcss, marker='o')
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()

km = KMeans(n_clusters=3, random_state=42, n_init=10)
print("Clusters:", km.fit_predict(X))


// 3 

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

X = np.array([
    [2,3], [3,4], [4,5], [5,6], [6,7],
    [7,8], [8,9], [9,10], [1,2], [2,4],
    [6,8], [7,9], [3,5], [8,10], [4,6]
])

y = np.array([0,0,0,0,1,1,1,1,0,0,1,1,0,1,0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

p = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test,p))
print("Accuracy:", accuracy_score(y_test,p))
print("Error Rate:", 1-accuracy_score(y_test,p))
print("Precision:", precision_score(y_test,p,zero_division=0))
print("Recall:", recall_score(y_test,p,zero_division=0))


// 4 

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *

X = np.array([
    [600, 30], [700, 40], [500, 25], [800, 50],
    [650, 35], [450, 20], [750, 45], [550, 28],
    [900, 55], [400, 18], [720, 42], [480, 22],
    [850, 52], [620, 32], [530, 26], [780, 48]
])

y = np.array([1,0,1,0,0,1,0,1,0,1,0,1,0,0,1,1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000,
                      random_state=42)

model.fit(X_train, y_train)
p = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,p))
print("Confusion Matrix:\n", confusion_matrix(y_test,p))

// 5 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

emails = [
    "Win free money now",
    "Congratulations you won a prize",
    "Claim your free gift",
    "Hello how are you",
    "Let's meet tomorrow",
    "Please send me notes",
    "Free lottery prize",
    "Call me later",
    "Win cash today",
    "How is your study"
]

y = [1,1,1,0,0,0,1,0,1,0]

X = CountVectorizer().fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

for model in [KNeighborsClassifier(n_neighbors=3), SVC()]:
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    print(type(model).__name__, "Accuracy:", accuracy_score(y_test,p))















