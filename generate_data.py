import numpy as np
import scipy.linalg as la

class Generate_tensor():
    def __init__(self, m, n, k=(1,1,1), sigma = 1):
        # m is the number of frontal slice inside the processus, data of size (m, n, n)
        # n is the full dimension of the tensor, (n, n, n)
        # k is the size of the signal (k1, k2, k3)
        self.m = m
        self.n = n
        self.k = k
        self.k1 = k[0]
        self.k2 = k[1]
        self.k3 = k[2]
        self.sigma = sigma

    def rank_one(self):
        if self.k[0] != 0:
            a = np.empty((self.k[0],self.k[1], self.k[2]))
            b = self.k[1]
            a.fill(self.sigma * 1/( np.sqrt( b**3 )) )
            X = np.zeros((self.m, self.n, self.n))
            X[:self.k[0],:,:][:,:self.k[1],:][:,:,:self.k[2]] = a
            return X + np.random.normal( 0, 1,  size = (self.m, self.n, self.n) )
        else :
            return np.random.normal( 0, 1,  size = (self.m, self.n, self.n) )


    def tensor_biclustering(self):
        # create a random vector 
        v = np.random.rand(self._m)
        v = v/la.norm(v)   # norm of v should be equal to 1

        J1_true = list(range(0,  self._k1))
        J2_true = list(range(0,  self._k2))

        # Generating signal tensor
        X = np.zeros((self._m,self._n1,self._n2))
        for i in J1_true:
            for j in J2_true:
                X[:,i,j] = (self._sigma1 / np.sqrt(self._k1*self._k2)) * v
        

        Z = np.random.normal(0,1, (self._m, self._n1, self._n2)) # random normal standard distribution N(0,1) with size (m, n1,n2)
           
        T = X + Z
        return T
        