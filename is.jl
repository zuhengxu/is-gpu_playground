using Distributions
using Random
using LinearAlgebra
using CUDA
using Flux
using LogExpFunctions: logsumexp


# 1. importance sampling
function is_log_weights(logq, logp, xs)
    _logws = logp(xs) .- logq(xs)
    logws = _logws .- logsumexp(_logws)
    return logws
end
function is_weights(logq, logp, xs)
    logws = is_log_weights(logq, logp, xs)
    return exp.(logws)
end

"""
    log_weights_normalization!(logws)
in place normalization of log weights
"""
function log_weights_normalization!(logws)
    logws .-= logsumexp(logws)
end

"""
    importancesampling(logq, logp, xs, f) 
importance sampling est of `E[f(x)]` given samples xs ∼ q

note: f(xs) should return a vector of shape (output_dim, N)
"""
function importancesampling(logq, logp, xs, f)
    ws = is_weights(logq, logp, xs)
    fs = f(xs)
    return fs * ws
end





# WARNING: bug in the following code
"""
    is_parallel_cpu(logq, logp, xs, f)
explicitly parallelize the weight calculation for each particle on CPU
"""
function is_parallel_cpu(logq, logp, xs, f)
    N = size(xs, 2)
    logws = fill(0f0, N) 

    Threads.@threads for i in 1:N
        # ERROR: logp and logq are high level functions, cannot be invoked in kernel
        @inbounds logws[i] = logp(@view(xs[:, i])) - logq(@view(xs[:, i])) 
    end
    
    log_weights_normalization!(logws)    
    logws .= exp.(logws) # in place exponentiation
    fs = f(xs)
    return fs * logws
end

"""
    _is_kernel!(logws, logp, logq, xs) 
cuda kernel for parallel unnormalized log-weights calculation for each particle
"""
function _is_kernel!(logws_gpu, logp, logq, xs_gpu, N)
    #  - `threadIdx().x` retrieves the x-index of the
    # current thread within its block. In CUDA, each thread within a block
    # has a unique index starting from 1, which can be used to determine
    # which part of the data the thread should operate on.
    #
    # - `i = threadIdx().x` assigns the thread index to the variable i, which is then used to index into the data.
    i = threadIdx().x 
    if i <= N
        @inbounds logws_gpu[i] = logp(xs_gpu[:, i]) - logq(xs_gpu[:, i])
    end
    return nothing
end


"""
    is_parallel_cuda_manual(logq, logp, xs_gpu, f; nthreads = 256) 
IS estimates using cuda kernel with manual thread configuration
"""
function is_parallel_cuda_manual(logq, logp, xs_gpu, f, nthreads)
    # CUDA.allowscalar(false)  # Disallow scalar operations on the GPU to ensure full parallelization
    N = size(xs_gpu, 2)
    logws_gpu = CUDA.zeros(Float32, N)  # Initialize log weights on GPU

    # cuda kernel of parallelized weight calculation
    CUDA.@sync begin
        @cuda threads=nthreads _is_kernel!(logws_gpu, logp, logq, xs_gpu, N)
    end

    # Normalize log weights and exponentiate
    log_weights_normalization!(logws_gpu)    
    logws_gpu .= exp.(logws_gpu) # in place exponentiation

    # Apply function `f` and multiply by log weights
    fs = f(xs_gpu)
    result = fs * logws_gpu  # matrix-vector multiplication assuming `fs` returns a matrix

    return CUDA.collect(result)  # Move the result back to CPU
end

"""
    is_parallel_cuda_auto(logq, logp, xs_gpu, f)
IS estimates using cuda kernel with automatic thread configuration
"""
function is_parallel_cuda_auto(logq, logp, xs_gpu, f)
    # CUDA.allowscalar(false)  # Disallow scalar operations on the GPU to ensure full parallelization
    N = size(xs_gpu, 2)
    logws_gpu = CUDA.zeros(Float32, N)  # Initialize log weights on GPU


    # using kernel configuration to automatically determine the number of threads and blocks
    kernel = @cuda launch=false _is_kernel!(logws_gpu, logp, logq, xs_gpu, N)
    config = launch_configuration(kernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)

    # cuda kernel of parallelized weight calculation
    CUDA.@sync begin
        kernel(logws_gpu, logp, logq, xs_gpu; threads, blocks)
    end

    # Normalize log weights and exponentiate
    log_weights_normalization!(logws_gpu)    
    logws_gpu .= exp.(logws_gpu) # in place exponentiation

    # Apply function `f` and multiply by log weights
    fs = f(xs_gpu)
    result = fs * logws_gpu  # matrix-vector multiplication assuming `fs` returns a matrix

    return result  # Move the result back to CPU
end






# 2. TODO: nonequilibrium IS (NEO)
# The importance sampling proposal is obatined by simulating multiple damped Hamiltonian trajectories (still has trackable density)
# reference: https://proceedings.neurips.cc/paper_files/paper/2021/file/8dd291cbea8f231982db0fb1716dfc55-Paper.pdf






# 3. TODO: AIS
