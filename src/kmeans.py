import numpy as np
from . import utils

def k_means(k: int, points: np.ndarray):
    rand = []
    for i in range(k):
        x = np.random.randint(0,len(points))
        rand.append(x)
    centroids = points[rand]
    
    
    new_centroids = np.zeros_like(centroids)
    centerpoints = np.zeros(len(points), dtype=int)
    if k == 1:
        return centerpoints,np.mean(points, axis=0)
    
    while np.linalg.norm(new_centroids - centroids) > 0.1 or np.all(centerpoints == 0):
        if not np.all(centerpoints == 0):
            new_centroids = centroids
            counter = 0
            for i in centroids:
                mask = centerpoints == counter
                if np.any(mask):
                    centroid = np.mean(points[mask], axis=0)
                    new_centroids[counter] = centroid
                counter += 1
        i = 0
        for point in points:
            
            smalldist = None
            counter2 = 0
            for centroid in centroids:
                dist = utils.distance(point, centroid)

                if smalldist is None or dist < smalldist:
                    smalldist = dist
                    centerpoints[i] = counter2
                counter2 += 1
            i += 1

    return centerpoints, centroids, points[rand]
