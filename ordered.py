# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:42:28 2024

@author: LinhNH
"""

import random
import numpy as np
import math

def euclidean_distance(p1, p2):
    if (p1 == 0) or (p2 == 0):
        return 0
    return np.linalg.norm(points[p1] - points[p2])

budget = 100
numpoints = 10
# size of bounding box
xmin = 0
xmax = 10
ymin = 0
ymax = 10

points = [(-1, -1)] # v_0
for i in range(numpoints):
    points.append(np.array([random.uniform(xmin, xmax), random.uniform(ymin, ymax)]))
    
search_cost = np.array([0], dtype = np.dtype(int))
target_distribution = np.array([0])
beta = np.array([0]) # false negative rates

# intitialize the problem

for i in range(numpoints):
    search_cost = np.append(search_cost, random.randint(1, 3))
    
for i in range(numpoints):
    target_distribution = np.append(target_distribution, random.uniform(0, 1))
    
for i in range(numpoints):
    beta = np.append(beta, random.uniform(0, 0.4))

sum_dist = sum(target_distribution)
target_distribution = target_distribution/sum_dist

# implementing this to the best of my understanding
big_C = 10
tau = math.ceil(budget*big_C)

# table of subproblems
p = np.zeros((numpoints+1, tau))
for i in range(1, numpoints+1):
    for t in range(tau):
        subproblems = np.array([])
        for k in range(0, i):
            for j in range(math.floor((t - euclidean_distance(i, k)*big_C)/(search_cost[i]*big_C)) + 1):
                subproblems = np.append(subproblems, p[k, max(0, t - math.floor(j*search_cost[i]*big_C + euclidean_distance(i, k)*big_C))] + (1 - beta[i]**j)*target_distribution[i])
            if (subproblems.size > 0):
                argmax = np.argmax(subproblems)
                p[i, t] = subproblems[argmax]

print(p[numpoints,tau-1])
