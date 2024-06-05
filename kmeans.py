import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_csv("Cust_Segmentation.csv")
df.head()

df=df.drop(["Customer Id","Defaulted","Address"],axis=1)
df=df.dropna(how='any')
df

k=KMeans(n_clusters=3, init='k-means++', n_init=12)

cen_ = k.fit(df)

center = cen_.cluster_centers_

plt.scatter(df.iloc[:, 3], df.iloc[:, 5], color="blue")
plt.scatter(center[:, 3], center[:, 5], color="red")
plt.show()

