using Distributions
using Random
using LinearAlgebra
using CUDA

CUDA.seed!(123)


# Gaussian target N(μ, Id) and proposal N(0, Id)
const log2π = Float32(log(2π))
function constructor(;dim = 2, device = "cpu", broadcast= true)
    μ = device=="cpu" ? 10ones(Float32,dim) : 10*cu(ones(Float32,dim))

    if broadcast
        # logp = x -> -dim*log2π/2 .- vec(sum(abs2, x.-μ; dims=1))./2
        logp = x -> -dim*log2π/2 .- vec(sum(abs2, x .- μ; dims=1))./2
        ∇logp = x -> μ .- x
        logq = x -> -dim*log2π/2 .- vec(sum(abs2, x; dims=1))./2
        ∇logq = x -> -x
    else
        logp = x -> -dim*log2π/2 .- sum(abs2, x.-μ)./2
        ∇logp = x -> μ .- x
        logq = x -> -dim*log2π/2 .- sum(abs2, x)./2
        ∇logq = x -> -x
    end    

    return logp, ∇logp, logq, ∇logq
end

function rng_device(device)
    return device == "cpu" ? Random.default_rng() : CUDA.default_rng()
end


# WARN: Distributions.jl doesn't play with CUDA.jl nicely, particularly the rand() function

# dim = 2
# μ = 10ones(Float32,dim)
# p_dist = MvNormal(μ, I)
#
# xs = rand(p_dist, 10) # on cpu
# xs_g = cu(xs) # on gpu
# ls = logpdf(p_dist, xs) # on cpu
# ls_g = logpdf(p_dist, xs_g) # on gpu ?
#
# # lets change stuff to cuda array
# μ_g = cu(μ)
# I_g = cu(Diagonal(ones(Float32,dim))) 
# p_dist_g = MvNormal(μ_g, I_g)
#
# rand(p_dist_g,10)
# ls_g = logpdf(p_dist_g, xs_g) # on gpu ?
#
# # manually implement the logpdf
# logp, _, logq, _ = constructor(;dim = dim, device="gpu")
# logp(xs_g) # actually on gpu
#
#
# # timing it
# dim = 1000
# μ = 10ones(Float32,dim)
# p_dist = MvNormal(μ, I)
# xs = rand(p_dist, 1000) # on cpu
# xs_g = cu(xs) # on gpu
# CUDA.@profile trace = true logp(xs_g)
# CUDA.@profile trace = true logq(xs_g)




