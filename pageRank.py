import numpy as np

def init(numNodes:int):
    x = np.ones((numNodes))/numNodes
    return x

def bruteForce(x: np.array, P: np.array, threshold: int):
    iter = 0
    min_iter = 0
    num_pages = x.shape[0] 
    rank = []
    for i in range(num_pages):
        rank.append([x[i]])
    # rank = np.append(rank, x)
    loop = True
    for i in range (100):
        x_new = np.matmul(P, x)
        iter += 1
        
        if np.linalg.norm(x_new - x, ord=1) < threshold:
            loop = False
            break
        x = x_new
        rank = np.append(rank, x)
    # for _ in range(100):
    #     x_new = np.matmul(P, x)
    #     for i in range(num_pages):
    #         rank[i].append(x_new[i])

    #     diff = x_new - x
    #     #print(np.linalg.norm(diff))
    #     if np.linalg.norm(diff) < threshold:
    #         iter = _
    #         break

    #     x = x_new

    return x, iter, rank


def eigenSolver(P:np.array)->np.array:
    eigenvalue, eigenvector = np.linalg.eig(P.T)
    dominant_value = np.argmax(eigenvalue.real)
    dominant_vector = (eigenvector[:,dominant_value]).real
    dominant_vector = dominant_vector/np.sum(dominant_vector)

    return dominant_vector


x = init(6)
P = np.array(((0, 0, 1/3, 0, 0, 0), (1/2, 0, 1/3, 0, 0, 0), (1/2, 1, 0, 0, 0, 0), (0, 0, 0, 0, 1/2, 1),
              (0, 0, 1/3 , 1/2, 0 , 0), (0, 0, 0, 1/2, 1/2, 0)))
print(P)
print(f"x = {x}")
# print(f"P = \n{P}")
x, min_iter, rank = bruteForce(x, P, 0.1)
print(f"x = {x}")
# print(len(iteration))
temp = np.linalg.eig(P.T)
# print(temp[0].real)
# print(min_iter)
# print(len(rank[0]))
v = eigenSolver(P)
print(v)