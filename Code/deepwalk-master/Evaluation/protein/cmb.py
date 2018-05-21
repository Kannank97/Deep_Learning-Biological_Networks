import csv
import pandas as pd

df1 = pd.read_csv('1024.csv')
df2 = pd.read_csv('p_ptm.csv')

df3 = pd.merge(df1,df2)	

df3.to_csv('multi_out.csv')