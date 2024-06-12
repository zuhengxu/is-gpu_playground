using Distributions
using Random
using LinearAlgebra
using CUDA
using LogExpFunctions: logsumexp
using BenchmarkTools

cd("example") # change working directory to benchmark/ if needed
include("../is.jl")
include("utils.jl")


# importance sampling can be naturally parallelized over particles, 
# so we can expect a linear speedup with respect to the number of computation units
# on CPU: we use @Threads.threads to parallelize the computation
# on GPU: we use CUDA.@cuda to launch a kernel on the GPU and parallelize the computation based on the number of threads and blocks

# cpu threads = 10

# let's work on 1d cases
# target p ∼ N(10, 1), proposal q ∼ N(0, 1)
logpdf_ratio_1d(x) = 50 - 10x
"""
    logws_cpu_parallel!(xs, logws)
parallelized log weights calculation for each particle on CPU
"""
function logws_cpu_parallel!(xs::AbstractVector{T}, logws) where T
    N = length(xs)
    Threads.@threads for i in 1:N
        @inbounds logws[i] = logpdf_ratio_1d(xs[i])
    end
end

function is_cpu_prallel(N)
    f = x -> x^2 # test function

    logws = fill(0f0, N) 
    xs = randn(Float32, N)
    
    logws_cpu_parallel!(xs, logws)
    
    log_weights_normalization!(logws)    
    logws .= exp.(logws) # in place exponentiation
    xs .= f.(xs)
    return logws' * xs
end  


"""
    _logws_cuda_parallel!(logws, xs, N)
cuda kernel for parallel log-weights calculation for each particle on GPU
"""
function _is_cuda_kernel!(logws, xs, N)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    # this for loop will later be parallelized over the grid of blocks and threads
    for i = index:stride:N
        @inbounds logws[i] = logpdf_ratio_1d(xs[i])
    end
    return nothing
end

function is_cuda_parallel(N; nthreads = 256)
    f = x -> x^2 # test function

    logws = CUDA.zeros(Float32, N)  # pre-allocate gpu memory for results
    xs = randn(CUDA.default_rng(), N)  # generate from proposal N(0, 1) on gpu

    # launching the kernel with enough threads:
    # it's important to configure blocks and threads according to gpu architecture.
    # for simplicity, launching one thread per data point, but in practice,
    # you may need to consider maximum threads per block and grid size.
    numblocks = ceil(Int, N/nthreads) # number of blocks based on the thread length
    CUDA.@sync begin
        # we manually configure the number of threads and blocks
        @cuda threads=nthreads blocks=numblocks _is_cuda_kernel!(logws, xs, N)
    end

    log_weights_normalization!(logws)    
    logws .= exp.(logws) # in place exponentiation
    xs .= f.(xs) # in place test function evaluation
    return logws' * xs
end

function is_cuda_parallel_auto(N)
    f = x -> x^2 # test function

    logws = CUDA.zeros(Float32, N)  # pre-allocate gpu memory for results
    xs = randn(CUDA.default_rng(), N)  # generate from proposal N(0, 1) on gpu

    # using kernel configuration to automatically determine the number of threads and blocks
    kernel = @cuda launch=false _is_cuda_kernel!(logws, xs, N)    
    config = launch_configuration(kernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)

    # cuda kernel of parallelized weight calculation
    CUDA.@sync begin
        kernel(logws, xs, N; threads, blocks)
    end
    log_weights_normalization!(logws)    
    logws .= exp.(logws) # in place exponentiation
    xs .= f.(xs)
    return logws' * xs
end


N = 10^8
@btime is_cpu_prallel(N)
@btime is_cuda_parallel(N; nthreads = 256)
@btime is_cuda_parallel_auto(N)










