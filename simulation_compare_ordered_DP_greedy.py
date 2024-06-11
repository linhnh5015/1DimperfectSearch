# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:52:52 2024

@author: LinhNH
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:42:28 2024

@author: LinhNH
"""

import random
import numpy as np
import math
import networkx as nx



def ordered_DP_heuristic(points, search_cost, target_distribution, beta, budget):
    
    def euclidean_distance(p1, p2):
        if (p1 == 0) or (p2 == 0):
            return 0
        return np.linalg.norm(points[p1] - points[p2])
    
    numpoints = len(points)
    # sort the points in a TSP (approximately) order
    geometric_graph = nx.Graph()
    for i in range(numpoints):
        geometric_graph.add_node(i, pos = points[i])

    for i in range(numpoints):
        for j in range(i + 1, numpoints):
            geometric_graph.add_edge(i, j, weight = euclidean_distance(i, j))
    overall_TSP_tour = nx.approximation.traveling_salesman_problem(geometric_graph)[:-1]

    points_temp = np.zeros((numpoints, 2))
    for i in range(numpoints):
        points_temp[i] = points[overall_TSP_tour[i]]
    points = np.copy(points_temp)
    
    v0 = np.array([-1,-1])
    points = np.append(v0, points)
    big_C = 10
    tau = math.ceil(budget*big_C)
    search_cost = np.append(np.array([0]), search_cost)
    target_distribution = np.append(np.array([0]), target_distribution)
    beta = np.append(np.array([0]), beta)


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
    return p[numpoints,tau-1]

def greedy_heuristic(points, search_cost, target_distribution, beta, budget):
    
    def euclidean_distance(p1, p2):
        return np.linalg.norm(points[p1] - points[p2])

    def updateProb(p, r):
        p_temp = np.copy(p)
        for j in range(len(p)):
            if j == r:
                p_temp[j] = beta[r]*p[r]/(beta[r]*p[r] + (1 - p[r]))
            else:
                p_temp[j] = p[j]/(beta[r]*p[r] + (1 - p[r]))
        return p_temp
    numpoints = len(points)
    s = np.zeros(numpoints)
    p = np.copy(target_distribution)
    tau = budget
    average_gain = [beta[i]*p[i]/search_cost[i] for i in range(numpoints)]
    r = np.argmax(average_gain)
    s[r] = s[r] + 1
    tau = tau - search_cost[r]
    p = updateProb(p, r)

    while tau > 0:
        average_gain = []
        for i in range(numpoints):
            if (search_cost[i] + euclidean_distance(r, i)) <= tau:
                average_gain.append(beta[i]*p[i]/(search_cost[i] + euclidean_distance(r, i)))
        if (len(average_gain) >= 1):
            r_prime = np.argmax(average_gain)
            tau = tau - search_cost[r_prime] - euclidean_distance(r, r_prime)
            p = updateProb(p, r_prime)
            s[r_prime] = s[r_prime] + 1
            r = r_prime
        else:
            tau = 0
        #print(s)
        
    prob_of_success = 0
    for i in range(numpoints):
        prob_of_success += (1 - beta[i]**s[i])*target_distribution[i]
    return prob_of_success

budget = 100
numpoints = 10
# size of bounding box
xmin = 0
xmax = 10
ymin = 0
ymax = 10


for budget in [100, 110, 120]:
    for numpoints in [10 + j for j in range(10)]:
        points = []
        for i in range(numpoints):
            points.append(np.array([random.uniform(xmin, xmax), random.uniform(ymin, ymax)]))
            
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
        greedy_p = greedy_heuristic(points, search_cost, target_distribution, beta, budget)
        ordered_p = ordered_DP_heuristic(points, search_cost, target_distribution, beta, budget)
        
        print('budget = ' + str(budget) + ', numpoints = ' + str(numpoints) + ', greedy gives ' + str(greedy_p) + ', ordered DP gives ' + str(ordered_p))
