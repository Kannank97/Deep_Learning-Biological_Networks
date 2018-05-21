import csv
import pandas as pd

df1 = pd.read_csv('cmblabels.csv')
df2 = pd.read_csv('vectors.csv')

df3 = pd.merge(df1,df2)
print(df3)

df3.to_csv('out.csv')