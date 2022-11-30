import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

def generate_gaussian_points(mu, sigma, num_points):
    points = np.random.multivariate_normal(mu, sigma, num_points)
    return points

mub = [20, 10]
sigmab = [[15, 1], [1, 15]]
num_pointsb = 100
pointsb = generate_gaussian_points(mub, sigmab, num_pointsb)

mur = [8, 8]
sigmar = [[15, 1], [1, 15]]
num_pointsr = 100
pointsr = generate_gaussian_points(mur, sigmar, num_pointsr)

plt.scatter(pointsb[:, 0], pointsb[:, 1])
plt.scatter(pointsr[:, 0], pointsr[:, 1])

plt.autoscale(enable=None, axis='both', tight=None)
plt.show()

pointsr = np.hstack((pointsr, np.zeros((num_pointsr, 1))))
pointsb = np.hstack((pointsb, np.ones((num_pointsb, 1))))


tr = np.vstack((pointsr, pointsb))

k = 5

def e_function(x0, y0, tr, k):
    distances = []
    
    for i in tr:
        dist = np.sqrt((x0-i[0])**2+(y0-i[1])**2)
        distances.append(dist)
    
    distances.sort()
    return distances[k-1]


def n_function(x0, y0, tr, k):
    epsilon = e_function(x0,y0,tr,k)
    dist_xy = []
    
    for i in tr:
        dis = np.sqrt((x0-i[0])**2+(y0-i[1])**2)
        if(dis <= epsilon):
            dist_xy.append((i[0], i[1], i[2]))
    
    return dist_xy


def f_function(x0, y0):
    li = n_function(x0,y0,tr,k)   
    count = Counter(li[:][2])
    max = count.most_common()[2][0]
    return max
    
    
label = f_function(15,10)
print(label)
