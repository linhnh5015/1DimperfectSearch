# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:26:38 2024

@author: LinhNH
"""

import random
import numpy as np
import math

budget = 200
numpoints = 6
xmin = -200
xmax = 200
min_search_cost = 1
max_search_cost = 5

points = np.array([], dtype = np.dtype(int))
search_cost = np.array([], dtype = np.dtype(int))
target_distribution = np.array([])
beta = np.array([]) # false negative rates

# intitialize the problem

while (points.size < numpoints):
    x = random.randint(xmin, xmax)
    if x not in points:
        points = np.append(points, random.randint(xmin, xmax))

points = np.sort(points)
    
for i in range(numpoints):
    search_cost = np.append(search_cost, random.randint(min_search_cost, max_search_cost))
    
for i in range(numpoints):
    target_distribution = np.append(target_distribution, random.uniform(0, 1))
    
for i in range(numpoints):
    beta = np.append(beta, random.uniform(0, 0.4))

sum_dist = sum(target_distribution)
target_distribution = target_distribution/sum_dist

p_prime = np.array([])
p = np.array([])
max_detection_prob = -1
for l in range(numpoints):
    for r in range(l, numpoints):
        budget_tau = max(0, budget - (points[r] - points[l]))
        p_prime = np.zeros(budget_tau)
        p = np.zeros(budget_tau)
        for i in range(l , r+1):
            for t in range(budget_tau):
                subproblems = np.array([])
                for j in range(math.floor(t/search_cost[i])):
                    subproblems = np.append(subproblems, p_prime[max(0, t - j*search_cost[i])] + (1 - beta[i]**j)*target_distribution[i])
                if (subproblems.size > 0):
                    argmax = np.argmax(subproblems)
                    p[t] = subproblems[argmax]
            p_prime = np.copy(p)
        if budget_tau >= 1:
            if p[budget_tau-1] > max_detection_prob:
                max_detection_prob = p[budget_tau-1]

print('The DP returns the maximum detection probability:' + str(max_detection_prob))
