import pandas as pd

data = pd.read_csv('/data/notebook_files/資料探勘/banana  (with class label).csv')  
data2 = pd.read_csv('/data/notebook_files/資料探勘/sizes3 (with class label).csv')  

missing_values_1 = data.isnull().sum()
missing_values_2 = data2.isnull().sum()
print("缺失值數：")
print(missing_values_1)
print(missing_values_2)
