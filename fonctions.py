from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from sklearn.preprocessing import normalize

# remove liste of slice in fixed dimension
def rem(donnee, dim, liste):
    while (liste != []):
        dimT = list(donnee.shape)
        if dim ==1:
            dimT[1] = dimT[1] - 1
            
            donneeI = np.zeros(dimT)
            for i in range(dimT[1]):
                if (i >= liste[-1]):
                    donneeI[:,1,:] = donnee[:,i+1,:]
                else:
                    donneeI[:,i,:] = donnee[:,i,:]
            donnee = donneeI  
            liste = liste[:-1]
        elif dim == 2:
            dimT[2] = dimT[2] - 1
            
            donneeI = np.zeros((dimT))
            for i in range(dimT[2]):
                if (i >= liste[-1]):
                    donneeI[:,:,i] = donnee[:,:,i+1]
                else:
                    donneeI[:,:,i] = donnee[:,:,i]
            donnee = donneeI  
            liste = liste[:-1]
            
        elif dim == 0:
            dimT[0] = dimT[0] - 1
            
            donneeI = np.zeros((dimT))
            for i in range(dimT[0]):
                if (i >= liste[-1]):
                    donneeI[i,:,:] = donnee[i+1,:,:]
                else:
                    donneeI[i,:,:] = donnee[i,:,:]
            donnee = donneeI  
            liste = liste[:-1]
                
    return donnee


# recovery rate
# I is the list the true cluster
# J is the list of estimated cluster
def recovery_rate(I, J):
    r_rate = 0
    for i in range(3):
        r  = set(I[i]).intersection(set(J[i]))
        r_rate += (len(r) / (3 * len(J[0])))
    return r_rate




def gound_truth_known_tensor_biclustering(true, estimation):  # true is a couple and the estimation as well
    # recovery rate
    r = len(set(true[1]).intersection(set(estimation[1]))) / (2*len(true[1]))
    r += len(set(true[0]).intersection(set(estimation[0]))) / (2*len(true[0]))
    return r


def find_adjusted_rand_score(vrai, estimation):
    result = 0
    for i in range(len(vrai)):
        result += adjusted_rand_score(vrai[i], estimation[i])
    return result/len(vrai)



def rayleigh_power_iteration(A, num_simulations: int, eps = 0.00000001):

    # initialization, a random vector
    a = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, a)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # Unit vector
        b_k = b_k1 / b_k1_norm

        # test the convergence
        if np.linalg.norm(b_k - a) <= eps :
            break
        else :
            a = b_k

    mu = np.dot(np.dot(b_k.T, A),b_k)

    return  mu, b_k



def normed_and_covariance(M):
    M = normalize(M, axis=0)
    M = (M.T).dot(M)
    return M


def initialize_c(Liste):   # Listte is the vector d in assending order
    i = 1
    c = abs(Liste[0] - Liste[1])
    for j in range(1, len(Liste)-1):
        if c < abs(Liste[j] - Liste[j+1]) :
            c = abs(Liste[j] - Liste[j+1])
            i = j

    return Liste[i]

def max_difference(valeur):
    if len(valeur)==1 :
        return 0
    c = []
    for i in range(len(valeur)):
        for j in range(i, len(valeur)):
            c.append(np.abs(valeur[i]-valeur[j]))
    return max(c)


def r(a):
    return round(a, 4)