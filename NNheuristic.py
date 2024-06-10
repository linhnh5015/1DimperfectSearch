# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:30:33 2024

@author: LinhNH
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import math

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def path_length(path):
    length = 0
    for i in range(1, len(path)):
        length += euclidean_distance(nodes[path[i]], nodes[path[i-1]])
    return length

def search(path, target, specific_num_passes):
    for i, node in enumerate(path):
        for j in range(specific_num_passes[i]):
            if node == target and random.random() > beta:
                return True
            
# size of bounding box
xmin = 0
xmax = 10
ymin = 0
ymax = 10

n = 100 # number of nodes
beta = 0.3 # false negative rate
budget = 50 # total budget for traveling and searching
search_cost = 1 # time it takes to do 1 search at a node

# generating random nodes in the plane
nodes = [np.array([random.uniform(xmin, xmax), random.uniform(ymin, ymax)]) for i in range(n)]

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
plt.axis('off')
for node in nodes:
    plt.plot(node[0], node[1], marker="o", markersize=1, markeredgecolor="black")
    
# starting at a random node
starting_point = target = random.randint(0, n-1)
# nearest neighbor greedy k-TSP
# find nearest neighbor of each node
 # i-th element is the nearest node to the ith node and the distance
k_TSP_paths = [] # greedy k-TSP paths for increasing k
for i in range(2, n):
    pivot = starting_point
    visited = [pivot]
    while len(visited) < i:
        distance_to_other_nodes = []
        for j in range(len(nodes)):
            if (j != pivot) and (j not in visited):
                distance_to_other_nodes.append([j, euclidean_distance(nodes[j], nodes[pivot])])
        next_node = distance_to_other_nodes[np.argmin([dist[1] for dist in distance_to_other_nodes])][0]
        visited.append(next_node)
        pivot = next_node
    k_TSP_paths.append(visited)

theoretical_probs = []
for p in k_TSP_paths:
    search_time = max(0, budget - path_length(p))
    total_num_passes = math.floor(search_time/search_cost)
    # figure out exactly how many passes at each node we make
    specific_num_passes = [total_num_passes // len(p) + (1 if x < total_num_passes % len(p) else 0)  for x in range (len(p))]
    theoretical_prob_of_detection = 0
    for s in specific_num_passes:
        theoretical_prob_of_detection += 1/n*(1 - beta**s)
    theoretical_probs.append(theoretical_prob_of_detection)
    
best_route_index = np.argmax(theoretical_probs)
best_route = k_TSP_paths[best_route_index]
print('Best route is: ' + str(best_route))
print('with probability of detection: ' + str(theoretical_probs[best_route_index]))

# place a target randomly at a node

search_time = max(0, budget - path_length(best_route))
total_num_passes = math.floor(search_time/search_cost)
specific_num_passes = [total_num_passes // len(best_route) + (1 if x < total_num_passes % len(best_route) else 0)  for x in range (len(best_route))]
detected = 0
for epcho in range(10000):
    target = random.randint(0, n-1)
    if search(best_route, target, specific_num_passes):
        detected += 1

print('simulated probability of detection: ' + str(detected/10000))
        
