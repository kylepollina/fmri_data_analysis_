from sklearn.neighbors import NearestNeighbors
import preprocessing

def run():
    X, y = preprocessing.run()
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    return distances, indices

