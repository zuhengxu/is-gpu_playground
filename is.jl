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
    importancesampling(logq, logp, xs, f) -> AbstractArray{Float32, 1}
importance sampling est of `E[f(x)]` given samples xs ∼ q

note: f(xs) should return a vector of shape (output_dim, N)
"""
function importancesampling(logq, logp, xs, f)
    ws = is_weights(logq, logp, xs)
    fs = f(xs)
    return fs * ws
end


# 2. TODO: nonequilibrium IS (NEO)
# The importance sampling proposal is obatined by simulating multiple damped Hamiltonian trajectories (still has trackable density)
# reference: https://proceedings.neurips.cc/paper_files/paper/2021/file/8dd291cbea8f231982db0fb1716dfc55-Paper.pdf






# 3. TODO: AIS
