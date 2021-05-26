import numpy as np
import pandas as pd
import sklearn.preprocessing as sc
from data import model
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

dataset_total = pd.concat([train_data["Open"], test_data["Open"]], axis=0)
inputs = dataset_total[len(dataset_total) - len(test_data) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 259):
   X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = model.predict(X_test)
predicted_stock_price = model.predict(X_test)