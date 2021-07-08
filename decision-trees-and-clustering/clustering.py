import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt

#obtain the datasets directly from UCI's website and import as dataframe
url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
dataframe = (pd.read_csv(url_train, delim_whitespace=True, header=None))

#drop the last column of data since they are not informative
X = dataframe.drop(dataframe.columns[7], axis=1)

#fit the dataset using Kmeans and using the elbow method to help us select k
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 11))     #k's range is set to try from 1 to 10
visualizer.fit(X)        # Fit the datseta to the visualizer
labels = visualizer.labels_     #save the labels to plot figure later
visualizer.show()       #show the results after implementing the elbow method

#plot the data into a 3D space
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X_reduced = PCA(n_components=3).fit_transform(X)    #since the dataset is a 7D dataset, we use PCA to reduce its dimention to 3D in able to plot 
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=labels, cmap=plt.cm.Set1)   #cmap=plt.cm.Set1 to give it some color rather than black and white
plt.show()      #show the results after plotting