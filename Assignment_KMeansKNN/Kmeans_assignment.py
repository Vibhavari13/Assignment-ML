
import numpy as np
import pandas as pd
import io
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


url = 'https://raw.githubusercontent.com/MSPawanRanjith/FileTransfer/master/kmean_dataset.csv'
s=requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))


# In[3]:


df.head()


# In[4]:


X=df.values
from sklearn.preprocessing import MinMaxScaler
X=sklearn.preprocessing.normalize(X)


# In[5]:


from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

def compute_bic(kmeans,X):

    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)





ks = range(1,10)

# run 9 times kmeans and save each result in the KMeans object
KMeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(X) for i in ks]

# now run for each cluster the BIC computation
BIC = [compute_bic(kmeansi,X) for kmeansi in KMeans]

print(BIC)
df = pd.DataFrame (BIC, columns = ['BIC values'],index=ks)
df.reset_index(drop=True,inplace=True)
df['Cluster_no']=ks
print(df.iloc[:,:])

plt.plot(ks,BIC,'r-o')
plt.title("kmeans_data  (cluster vs BIC)")
plt.xlabel("# clusters")
plt.ylabel("# BIC")
plt.show()


# In[6]:


def find_local_max(o):
    m=argrelextrema(o, np.greater)
    l=[]
    for i in m:
        l.append(i)
    print("Local maxima :",l)
    print("The maximum BIC value is at cluster number",max(*l)+1) 
    
import numpy as np
from scipy.signal import argrelextrema
o=np.array(BIC)

find_local_max(o)


      
    


# In[7]:


from sklearn.cluster import KMeans


# In[ ]:





# In[8]:


kmeans3 = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
x1=kmeans3.fit_predict(X)
labels3 = kmeans3.labels_


# In[9]:


estimator = KMeans(n_clusters=3)
estimator.fit(X)


# In[10]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 0], X[:, 1], X[:, 2],
c=labels3.astype(np.float), edgecolor="k", s=50)

ax.set_xlabel("first column")
ax.set_ylabel("second column")
ax.set_zlabel("third column")
plt.title("K Means", fontsize=14)
plt.show()


# In[11]:


centroids = kmeans3.cluster_centers_
print('The centroid values are as follows:')
print(centroids)

plt.scatter(X[:,0],X[:,1],c=X[:,2],cmap='rainbow')
plt.scatter(centroids[:,0] ,centroids[:,1], color='black', marker='x')

plt.title('2-d representation')
plt.xlabel('Column 1')
plt.ylabel('Column 2')
plt.show()



# In[12]:


centroids = kmeans3.cluster_centers_

fig = plt.figure(1, figsize=(7,7))

ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(centroids[:,0] ,centroids[:,1],centroids[:,2], color='red', marker='X')
ax.scatter(X[:, 0], X[:, 1], X[:, 2],
c=labels3.astype(np.float), edgecolor="k", s=50)


ax.set_xlabel("first column")
ax.set_ylabel("second column")
ax.set_zlabel("third column")

plt.title("K Means", fontsize=14)
plt.show()


# In[ ]:





