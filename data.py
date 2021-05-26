import pandas as pd
data  =  pd.read_csv("appl.csv")
data["Date"]=pd.to_datetime(data.Date)


#Formatting Data w/out $$

data["Open"] = data["Open"].str.replace("\$", "",regex = True).astype(float)
data["High"] = data["High"].str.replace("\$", "",regex = True).astype(float)
data["Close/Last"] = data["Close/Last"].str.replace("\$", "",regex = True).astype(float)
data["Low"] = data["Low"].str.replace("\$", "",regex = True).astype(float)

#Formatting Columns

data.rename(columns= {" Close/Last" : "Close/Last", " Volume" : "Volume",
 " Open" : "Open", " High" : "High", " Low" : "Low"}, inplace = True) 

#Split into train and test:
data_to_train = data[:1000]
data_to_test = data[1000:]

data_to_train.to_csv("train_data.csv")
data_to_test.to_csv("test_data.csv")


print(data.head())