using Distributions
using Random
using LinearAlgebra
using CUDA
using LogExpFunctions: logsumexp

cd("example") # change working directory to benchmark/ if needed
include("../is.jl")
include("utils.jl")
include("setup.jl")



# importance sampling can be naturally parallelized over particles, 
# so we can expect a linear speedup with respect to the number of computation units
# on CPU: we use @Threads.threads to parallelize the computation
# on GPU: we use CUDA.@cuda to launch a kernel on the GPU and parallelize the computation based on the number of threads and blocks

# cpu threads = 10

# benchmark importance sampling on cpu and gpu
function is_parallel_timing(device, dim, n; n_run=10, nthreads = 256)
    f = x -> x # NOTE: we choose the test function corresponding to the mean estimate
    logp, _, logq, _ = constructor(; dim=dim, device=device, broadcast=false)
    xs = device=="cpu" ? randn(Float32, dim, n) : cu(randn(Float32, dim, n))

    # just execute the function to nrun times to track the time
    if device == "cpu"
        args = (logq, logp, xs, f)
        time_log = noob_timing(is_parallel_cpu, args...; n_run=n_run)
    elseif device == "gpu_manual"
        args = (logq, logp, xs, f, nthreads)
        # ERROR: logp and logq are high level functions, cannot be invoked in kernel
        time_log = noob_timing(is_parallel_cuda_manual, args...; n_run=n_run)
    else
        args = (logq, logp, xs, f)
        # ERROR: logp and logq are high level functions, cannot be invoked in kernel
        time_log = noob_timing(is_parallel_cuda_auto, args...; n_run=n_run)
    end

    return time_log
end

dim = 100
n_particle = 10000
ts_cpu = is_parallel_timing("cpu", dim, n_particle; n_run=10)
ts_gpu_manual = is_parallel_timing("gpu_manual", dim, n_particle; n_run=10, nthreads = 256)
ts_gpu_auto = is_parallel_timing("gpu_auto", dim, n_particle; n_run=10)


# function _is_kernel!(logws_gpu, logp, logq, xs_gpu, N)
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = gridDim().x * blockDim().x
#     for i = index:stride:N
#         @inbounds logws_gpu[i] = logp(@view(xs_gpu[:, i])) - logq(@view(xs_gpu[:, i]))
#     end
#     return 
# end
#
# # Example of how to invoke this kernel function:
# function run_kernel(logp, logq, n)
#     # assuming `logp` and `logq` are suitable gpu functions and `xs` is already on gpu
#     logws_gpu = CUDA.zeros(Float32, n)  # pre-allocate gpu memory for results
#     xs_gpu = cu(randn(Float32, dim, n))  # ensure data is on gpu
#
#     # launching the kernel with enough threads:
#     # it's important to configure blocks and threads according to gpu architecture.
#     # for simplicity, launching one thread per data point, but in practice,
#     # you may need to consider maximum threads per block and grid size.
#     numblocks = ceil(Int, n/256)
#     xs_gpu = cu(randn(Float32, dim))  # ensure data is on gpu
#     CUDA.@sync begin
#         @cuda threads=256 blocks=numblocks dynamic = true _is_kernel!(logws_gpu, logp, logq, xs_gpu, n)
#     end
#
#     # optionally, copy results back to cpu memory if needed elsewhere
#     logws = CUDA.collect(logws_gpu)
#     return logws
# end
