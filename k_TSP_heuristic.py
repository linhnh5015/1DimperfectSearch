# -*- coding: utf-8 -*-
"""
Created on Tue May 21 22:23:16 2024

@author: LinhNH
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:30:33 2024

@author: LinhNH
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import math
import networkx as nx

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

n = 20 # number of nodes
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
    
# build complete geometric group to do TSP
geometric_graph = nx.Graph()
for i in range(n):
    geometric_graph.add_node(i, pos = nodes[i])

for i in range(n):
    for j in range(i + 1, n):
        geometric_graph.add_edge(i, j, weight = euclidean_distance(nodes[i], nodes[j]))
starting_point = random.randint(0, n-1)
overall_TSP_tour = nx.approximation.traveling_salesman_problem(geometric_graph)[:-1]
index_of_starting_point = overall_TSP_tour.index(starting_point)

shifted_starting_point_to_first = [overall_TSP_tour[(index_of_starting_point + i) % len(overall_TSP_tour)] for i in range(len(overall_TSP_tour))]
# for i in range(len(overall_TSP_tour)):
#     x_start = nodes[overall_TSP_tour[i]][0]
#     y_start = nodes[overall_TSP_tour[i]][1]
#     x_end = nodes[overall_TSP_tour[(i+1)%len(overall_TSP_tour)]][0]
#     y_end = nodes[overall_TSP_tour[(i+1)%len(overall_TSP_tour)]][1]
#     ax.annotate('',xy=(x_start,y_start),xytext=(x_end,y_end),arrowprops=dict(arrowstyle= '-',
#                              color='black', lw=0.4, ls='-'))

k_TSP_paths = [[starting_point]]
for k in range(2, n):
    k_tsp_path = shifted_starting_point_to_first[:k]
    k_tsp_path = nx.approximation.traveling_salesman_problem(geometric_graph, nodes = k_tsp_path)[:-1]
    k_TSP_paths.append(k_tsp_path)
        

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
