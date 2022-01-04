from sklearn.cluster import DBSCAN
from sklearn import metrics

# DBSCANMap class
# Properties:
"""
n_clusters : number of clusters
n_noise : number of noise points
labels: array of labels of clusters matching the index of original data
silhouette_coef: a score metric, not very sure what it does tbh
data: pandas dataframe of lat long data with labels
centroids: 2d list of centroids of individual clusters
clusterMap: Folium map on clusters
noiseMap: Folium map on noise
"""

class DBSCANMap:
    def __init__(
            self,
            n_clusters,
            n_noise,
            labels,
            silhouette_coef,
            data,
            centroids,
            clusterMap,
            noiseMap
            ):
        self.n_clusters = n_clusters
        self.n_noise = n_noise
        self.labels = labels
        self.silhouette_coef = silhouette_coef
        self.data = data
        self.centroids = centroids
        self.clusterMap = clusterMap
        self.noiseMap = noiseMap


# Function to apply DBSCAN to data and return clusters and noise
# Params:
"""
data: long lat data in pandas dataframe
eps: epsilon in float
min_samples: minimum samples to define a cluster in int
"""
# Returns:
"""
DBSCANMap
"""
def plot_DBSCAN(data, eps, min_samples):
    X = np.array(data, dtype = 'float64')
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    silhouette_coef = metrics.silhouette_score(X, labels)


    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    labelledData = data.insert(2, "labels")

    # Plot clusters
    clusters = labelledData[labelledData["labels"] != -1][["Latitude", "Longitude"]]

    clusterMap = folium.Map(singaporeLatlong, zoom_start=11)
    clusterMap.add_child(plugins.HeatMap(clusters, radius=14))

    # Find center of clusters by getting the mean, this assumes the points are close enough that the earth is planar

    centroids = []
    for i in range(n_clusters_):
        df = labelledData[labelledData["labels"] == i][["Latitude", "Longitude"]]
        centroid = [df.mean().Latitude, df.mean().Longitude]

        centroids.append(centroid)
        folium.Markere(centroid, popup=f'<b>Cluster {i}</b>').add_to(clusterMap)


    # Plot Outliers/Noise, this can be considered unusual idling
    noise = labelledData[labelledData["labels"] == -1][["Latitude", "Longitude"]]

    noiseMap = folium.Map(singaporeLatlong, zoom_start=11)
    def plotCircle(row):
        folium.CircleMarker(location=[row.Latitude, row.Longitude],
                            radius=4,
                            weight = 5).add_to(noiseMap)

    noise.apply(plotCircle, axis=1)


    return DBSCANMap(
            n_clusters_,
            n_noise_,
            labels,
            silhouette_coef,
            labelledData,
            centroids,
            clusterMap,
            noiseMap
            )
