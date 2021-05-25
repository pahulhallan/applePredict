import pandas as pd
data  =  pd.read_csv("appl.csv")
print(data.head())
data["Date"]=pd.to_datetime(data.Date)
data.head()

data[‘Open’] = data[‘Open’].replace({"\$":’’}, 
regex=True).astype(float)

'''
data[‘High’] = data[‘High’].replace({"\$": ‘’}, regex=True).astype(float)
data[‘Close/Last’] = data[‘Close/Last’].replace({"\$": ‘’}, regex=True).astype(float)
data[‘Low’] = data[‘Low’].replace({"\$": ‘’}, regex=True).astype(float)
'''