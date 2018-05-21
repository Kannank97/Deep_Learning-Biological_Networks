import csv
import pandas as pd

df1 = pd.read_csv('multi_out.csv')
df2 = pd.read_csv('p_arch.csv')

df3 = pd.merge(df1,df2)	

df3.to_csv('128v.csv')