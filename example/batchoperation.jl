using Distributions
using Random
using LinearAlgebra
using CUDA
using LogExpFunctions: logsumexp

cd("example") # change working directory to benchmark/ if needed
include("../is.jl")
include("utils.jl")
include("setup.jl")

# benchmark importance sampling on cpu and gpu
function is_timing(device, dim, n; n_run=10)
    f = x -> x # NOTE: we choose the test function corresponding to the mean estimate
    RNG = rng_device(device)
    logp, _, logq, _ = constructor(; dim=dim, device=device)
    xs = randn(RNG, dim, n)

    args = (logq, logp, xs, f)
    # just execute the function to nrun times to track the time
    time_log = noob_timing(importancesampling, args...; n_run=n_run)

    return time_log
end

function is_dim_scaling(device, dims, n; n_run=10)
    time_logs = zeros(n_run, length(dims))

    for (i, dim) in enumerate(dims)
        time_log = is_timing(device, dim, n; n_run=n_run)
        time_logs[:, i] = time_log
    end
    return time_logs
end
function is_dim_scaling(dims, n; n_run=10)
    ts_dim_cpu = is_dim_scaling("cpu", dims, n; n_run=n_run)
    ts_dim_gpu = is_dim_scaling("gpu", dims, n; n_run=n_run)
    return ts_dim_cpu, ts_dim_gpu
end

function is_particle_scaling(device, dim, Ns; n_run=10)
    time_logs = zeros(n_run, length(Ns))

    for (i, n) in enumerate(Ns)
        time_log = is_timing(device, dim, n; n_run=n_run)
        time_logs[:, i] = time_log
    end
    return time_logs
end
function is_particle_scaling(dim, Ns; n_run=10)
    ts_particle_cpu = is_particle_scaling("cpu", dim, Ns; n_run=n_run)
    ts_particle_gpu = is_particle_scaling("gpu", dim, Ns; n_run=n_run)
    return ts_particle_cpu, ts_particle_gpu
end

################
# benchmarking (let's see some figures!)
###############
N = 1000
dims = [10^i for i in 1:6]
ts_dim_cpu, ts_dim_gpu = is_dim_scaling(dims, N; n_run=10)

d = 100
Ns = [10^i for i in 1:6]
ts_particle_cpu, ts_particle_gpu = is_particle_scaling(d, Ns; n_run=10)

# use unicodeplots backend if we want to plot in terminal
# unicodeplots()

if !isfile("figure")
    mkdir("figure")
end

pl_dim = ribbon_plot(
    ts_dim_cpu,
    ts_dim_gpu,
    dims;
    xlabel="dim of targets",
    ylabel="time (s)",
    title="dimension scaling with fixed n(#ptcs = $N)",
)
savefig(pl_dim, "figure/dim_scaling.png")

pl_pts = ribbon_plot(
    ts_particle_cpu,
    ts_particle_gpu,
    Ns;
    xlabel="N particles",
    ylabel="time (s)",
    title="# particle scaling with fixed dim(dim = $d)",
)
savefig(pl_pts, "figure/particle_scaling.png")
