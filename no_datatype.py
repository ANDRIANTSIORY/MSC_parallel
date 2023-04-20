#!/usr/bin/env python3

# without using derived datatype

# Arguments : 
    # 1 - the cluster size inside the data
    # 2 - gamma (weight of the signal)
    # 3 - epsilon (hyperparameter)
    # 4 - dimension of the data mode-1       (time in the figure)
    # 5 - dimension of the data mode-2       (individuals in the figure)
    # 6 - dimension of the data mode-3       (features in the figure)

import numpy as np
from mpi4py import MPI
import sys
import math
import fonctions as f
import scipy.linalg as la
import os.path
import time
import generate_data as gdata

# load the data file
comm = MPI.COMM_WORLD
commRank = MPI.COMM_WORLD.Get_rank()
commSize = MPI.COMM_WORLD.Get_size()

color = math.floor(  3 * MPI.COMM_WORLD.Get_rank() / MPI.COMM_WORLD.Get_size() )
colorComm = MPI.COMM_WORLD.Split( color, MPI.COMM_WORLD.Get_rank() )

inter_color =  MPI.COMM_WORLD.Get_rank() % colorComm.Get_size()    # the number of files
Rcomm = MPI.COMM_WORLD.Split( inter_color, MPI.COMM_WORLD.Get_rank() )

color_rank = colorComm.Get_rank()
color_size = colorComm.Get_size()


# -----------------------------------------
dim = int(sys.argv[-3])   # dimension zero of all data tensor
d_shape = (dim, dim, dim)
# find the last processes
if (color_rank != (color_size - 1)) :
    dim0 = int(dim / color_size)
else :
    dim0 = int(dim - (int(dim/color_size) * (color_size-1)))
#------------------------------------------

Rayleigh = False
eps = float(sys.argv[-4])
gamma = float(sys.argv[-5])
card = int(sys.argv[-6])


dim_file = (dim0, int(sys.argv[-2]), int(sys.argv[-1]))

#----Start chrono --------
t_data_generator = time.time()

# Each color load the data splitted along frontal slice

if (color_rank+1) * dim0 <= card:
    data = gdata.Generate_tensor(dim0,dim,k=( dim0,card, card), sigma=gamma) 
elif (color_rank+1)*dim0 > card and (card - (color_rank*dim0) > 0 ):
    data = gdata.Generate_tensor(dim0,dim,k=( card - (color_rank*dim0),card, card), sigma=gamma) 
else :
    data = gdata.Generate_tensor(dim0,dim,k=( 0,0,0), sigma=gamma) 
data = data.rank_one()

t_data_generator = time.time() - t_data_generator
t_data_sharing = time.time()


# move axis inside the tensor
if color == 1:
    data = np.moveaxis(data, 2, 0)
if color == 2:
    data = np.moveaxis(data, 1, 0)
data = np.ascontiguousarray(data, dtype=np.float64)

# Scatterv, gather the frontal slice in each 
l = int(dim / color_size)


# The main slice need to be in the frontal slice
# color 1 look for cluster in the second dimension
# color 2 look for cluster in the third dimension
if color != 0:
    data_liste = []
    for i in range(color_size):     # we take the processes in color 1, one by one
        if color_rank == i :
            dim_file = np.array(list(data.shape))   # the axis in the data already changed
        else :
            dim_file = np.empty(3, dtype='int')
        colorComm.Bcast(dim_file, root = i)

        sendcountes, displacements, thread = [], [], []
        for j in range(color_size):
            if j < color_size -1 :
                thread.append(int(dim_file[0]/color_size))
                sendcountes.append(dim_file[1]*dim_file[2]* int(dim_file[0]/color_size))
            else:
                thread.append( dim_file[0] - np.sum(thread) )
                sendcountes.append(dim_file[1]*dim_file[2]* (dim_file[0] - np.sum(thread[:-1])))
            displacements.append(np.sum(sendcountes[:j]))

        x = np.zeros(( thread[color_rank] , dim_file[1] , dim_file[2] ))
        colorComm.Scatterv([data, tuple(sendcountes), tuple(displacements), MPI.DOUBLE], x, root=i)
        data_liste.append(x)

    # concatenate the data_liste to have the different slices
    # use the same variable data 
    data = np.concatenate(data_liste, axis = 1)
    # free the data_liste
    del data_liste
    #the slice of the dimension 1 is a frontal slice of the data

# -----useless commanc---wait for the other color---
t_data_sharing = time.time() - t_data_sharing
#p = comm.allreduce(time.time(),op=MPI.MAX) - comm.allreduce(time.time(),op=MPI.MIN)   # print(p)
t_computation = time.time()

# compute the covariance matrix of each slice in each processes
shape0 = data.shape[0]
Lambda = []
for k in range(shape0):
    one_slice =  f.normed_and_covariance(data[k,:,:])

    if Rayleigh == True:
        w, v = f.rayleigh_power_iteration(one_slice, 300)
    else :
        w, v = la.eig(one_slice)
        w, v = w[0], v[:,0]
        
    w, v =  w.real, v.real

    if k == 0 :
        V_ = (w*v).reshape(1,-1)
    else :
        V_ = np.vstack((V_, (w*v).reshape(1,-1)))

    Lambda.append(w)

Lambda_max = np.array(0.0, 'd')
Lambda = np.max(Lambda)
colorComm.Allreduce(Lambda, Lambda_max, op=MPI.MAX)

# Normalize the matrix V
V_ = V_ / Lambda_max
dim_file = (int(sys.argv[-3]), int(sys.argv[-2]), int(sys.argv[-1]))

#V_normalize_ = np.zeros((V_.shape[0]*color_size, V_.shape[1]))    # begger matrix, after, we remove the rows zero
V_normalize_ = np.zeros((dim_file[(2 + color - color %2)%3], V_.shape[1]))

# sendcountes, gather each nomber of elements in each processes
count = V_.shape[0] * V_.shape[1]
count_d = V_.shape[0]

# gather count to the rank=0 of each color
sendcountes = colorComm.allgather(count)
sendcountes_d = colorComm.allgather(count_d)



displacements, displacements_d  = [], []

for k in range(color_size):
    displacements.append(np.sum(sendcountes[:k]))
    displacements_d.append(np.sum(sendcountes_d[:k]))

colorComm.Allgatherv([V_, MPI.DOUBLE], [V_normalize_, sendcountes, displacements, MPI.DOUBLE])   # Allgather(V_, V_normalize_ )
# remove zero rows in V_normalize
# remove rows having all zeroes
#test_sim_matrix = V_normalize_.dot(V_normalize_.T)
#test_sim_matrix = np.abs(test_sim_matrix)

v = (V_).dot(V_normalize_.T)       # V^t V
v = np.abs(v)                        # the absolute value

# combine the similarity matrix
sim_matrix = colorComm.gather(v, root = 0)
if color_rank == 0:
    sim_matrix = np.concatenate(sim_matrix)  # store each similarity matrix (one mode) in each rank 0
                                                # np.concatenate() , default , axis = 0

vector = np.sum(v, axis = 1)                       # sum of each line

# gather the result
d = None
if color_rank == 0 :
    d = np.empty(V_normalize_.shape[0] , dtype='d')  

# use gatherv for the reconstruction of the vector d
colorComm.Gatherv([vector, MPI.DOUBLE], [d, sendcountes_d, displacements_d, MPI.DOUBLE], root=0)

# time for the sequential part

# Focus on the group ( rank 0 in each local_comm)
if color_rank == 0 : 

    V_order = d.copy()  
    V_order.sort()
    gap = f.initialize_c(V_order) 

    thm1_indice = [k for k, l in enumerate(d) if l >= gap]   # d is the original vector

    thm1_value  = [d[k] for k in thm1_indice]    # The value for all indices in thm1 

    thm = thm1_indice.copy()
    c = len(thm1_indice)   
    if (eps < 1 / ((len(d) - c)**2)) :   # with a fix epsilon
        while ( (f.max_difference(thm1_value) > (c * (eps / 2)) + (np.log(len(d)-c)**0.5)) ):
            thm = [k for k, l in enumerate(d) if l > min(thm1_value)]
            thm1_value.remove(min(thm1_value))
            c -= 1



t_computation = time.time() - t_computation
    

t_data_sharing = comm.allreduce(t_data_sharing,op=MPI.MAX)   # print(p)


# mean of the similarity

if color_rank == 0:

    similarity_mean = np.mean(sim_matrix[thm,:][:,thm])
    similarity  = Rcomm.gather(similarity_mean, root=0)    # ok
    cluster = Rcomm.gather(thm, root=0)


if comm.Get_rank()  == 0:
    nbr_processor = comm.Get_size()
    real_cluster = [ [i for i in range(card)], [i for i in range(card)], [i for i in range(card)] ]
    rec_rate = f.recovery_rate(cluster, real_cluster)

    res = [card, eps, nbr_processor, d_shape, gamma , rec_rate, similarity, f.r(t_data_generator), f.r(t_data_sharing) ,f.r(t_computation) ]
    if os.path.isfile("./result_datatype.txt"):
        # append the result to the file
        with open('./result_datatype.txt', 'a') as f:
            f.write(str(res) + "\n")

    else:
        with open('./result_datatype.txt', 'w') as f:
            f.write(str(res) + "\n")






