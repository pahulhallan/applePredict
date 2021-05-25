import pandas as pd
data  =  pd.read_csv("appl.csv")
print(data.head())
data["Date"]=pd.to_datetime(data.Date)
