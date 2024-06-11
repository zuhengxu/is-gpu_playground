# importance sampling can be easily parallelized over particles, 
# so we can expect a linear speedup with respect to the number of computation units
# on CPU: we use @Threads.threads to parallelize the computation
# on GPU: we use CUDA.jl to parallelize the computation



