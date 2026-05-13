import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r"C:\Users\SOORAJ R NAIR\OneDrive\Desktop\assignment\genz_social_media_usage_1M.csv"
df = pd.read_csv(file_path)
print(df.head())
print(df.info()) 
print(df.shape)
print(df.describe())


df['age'].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

df['addiction_level'].value_counts().plot(kind='bar')
plt.title("Addiction Level Distribution")
plt.show()
