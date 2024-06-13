import random
import numpy as np

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

budget = 100
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

sum_dist = sum(target_distribution)
target_distribution = target_distribution/sum_dist #normalize

for i in range(numpoints):
    beta = np.append(beta, random.uniform(0, 0.4))

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
    print(s)
    
prob_of_success = 0
for i in range(numpoints):
    prob_of_success += (1 - beta[i]**s[i])*target_distribution[i]
    
print(prob_of_success)
