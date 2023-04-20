
#!/bin/bash

# Arguements :
	# 1 - cluster size
	# 2 - signal strensth
	# 3 - epsilon
	# 4 - dimension of the tensor (dim0 = dim2 = dim3) 

mpiexec -n 6 python3 msc_parallel.py $1 $2 $3 $4 $4 $4
