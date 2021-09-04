import pandas as pd
import numpy as np
from numpy import log, dot, e
from numpy.random import rand
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# x = [[1, 2], [3, 4]]
# x = np.array(x)
# weight = [4, 1]

# print("x=", x)
# print(x.T)
# print("weight=", weight)

# dot1 = np.dot(x, weight)
# dot2 = dot(x.T, weight)

# print(dot1)
# print(dot2)

X = load_breast_cancer()['data']
y = load_breast_cancer()['target']
feature_names = load_breast_cancer()['feature_names'] 


buh = pd.DataFrame(np.concatenate((X, y[:, None]), axis=1), columns=np.append(feature_names, 'Target')).head()
print(buh)

scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)
print(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

class LogisticRegression:
    
    def sigmoid(self, z): return 1 / (1 + e**(-z))
    
    def cost_function(self, X, y, weights):                 
        z = dot(X, weights)        
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))        
        return -sum(predict_1 + predict_0) / len(X)
    
    def fit(self, X, y, epochs=25, lr=0.05):  
        loss = []
        weights = rand(X.shape[1])
        N = len(X)

        print("X")      
        print(X) 
        print("weight")      
        print(weights)          
        for _ in range(epochs):        
            # Gradient Descent
            print("dot")
            print(dot(X, weights))
            print("buh")
            y_hat = self.sigmoid(dot(X, weights))
            # print(y_hat)
            weights -= lr * dot(X.T,  y_hat - y) / N            
            # Saving Progress
            loss.append(self.cost_function(X, y, weights)) 
            
        self.weights = weights
        print(weights)
        self.loss = loss
    
    def predict(self, X):        
        # Predicting with sigmoid function
        z = dot(X, self.weights)
        # Returning binary result
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]

logreg = LogisticRegression()
logreg.fit(X_train, y_train, epochs=1, lr=0.5)
y_pred = logreg.predict(X_test)

# print(classification_report(y_test, y_pred))
# print('-'*55)
# print('Confusion Matrix\n')
# print(confusion_matrix(y_test, y_pred))