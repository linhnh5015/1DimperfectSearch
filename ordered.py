# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:42:28 2024

@author: LinhNH
"""

import random
import numpy as np
import math
import networkx as nx

def euclidean_distance(p1, p2):
    if (p1 == -1) or (p2 == -1):
        return 0
    return np.linalg.norm(points[p1] - points[p2])

budget = 50
numpoints = 10
# size of bounding box
xmin = 0
xmax = 10
ymin = 0
ymax = 10

points = [np.array([random.uniform(xmin, xmax), random.uniform(ymin, ymax)]) for i in range(numpoints)]    

search_cost = np.array([], dtype = np.dtype(int))
target_distribution = np.array([])
beta = np.array([]) # false negative rates

# intitialize the problem

for i in range(numpoints):
    search_cost = np.append(search_cost, random.randint(1, 3))
    
for i in range(numpoints):
    target_distribution = np.append(target_distribution, random.uniform(0, 1))
    
for i in range(numpoints):
    beta = np.append(beta, random.uniform(0, 0.4))

sum_dist = sum(target_distribution)
target_distribution = target_distribution/sum_dist

# build complete geometric group to do TSP
geometric_graph = nx.Graph()
for i in range(numpoints):
    geometric_graph.add_node(i, pos = points[i])

for i in range(numpoints):
    for j in range(i + 1, numpoints):
        geometric_graph.add_edge(i, j, weight = euclidean_distance(i, j))

# random starting point that defines p(0, tau) in pseudocode
starting_point = random.randint(0, numpoints-1)
overall_TSP_tour = nx.approximation.traveling_salesman_problem(geometric_graph)[:-1]
index_of_starting_point = overall_TSP_tour.index(starting_point)
shifted_starting_point_to_first = [overall_TSP_tour[(index_of_starting_point + i) % len(overall_TSP_tour)] for i in range(len(overall_TSP_tour))]

# implementing this to the best of my understanding
big_C = 10
tau = math.ceil(budget*big_C)

# table of subproblems
p = np.zeros((numpoints, tau))
for i in range(numpoints):
    for t in range(tau):
        subproblems = np.array([])
        for k in range(-1, i):
            for j in range(math.floor((t - euclidean_distance(i, k))/search_cost[i]) + 1):
                subproblems = np.append(subproblems, p[k, max(0, t - math.floor(j*search_cost[i]*big_C + euclidean_distance(i, k)*big_C))] + (1 - beta[i]**j)*target_distribution[i])
            if (subproblems.size > 0):
                argmax = np.argmax(subproblems)
                p[i, t] = subproblems[argmax]

print(p[numpoints-1,tau-1])