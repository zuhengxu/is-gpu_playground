using TickTock, Suppressor, ProgressMeter
using StatsBase
using Plots


#######################
## timing
#######################

# Callback niceties
call(f::Function, args...) = f(args...)
call(f::Function, args::Tuple) = f(args...)


"""
    noob_timing(f::Function, args...; n_run = 100)

benchmark function to evaluate time by calling `f(args...)` for `n_run` times 
and return the time for each run 
"""
function noob_timing(f, args...; n_run = 100)
    time_log = zeros(n_run+1)
    count = 0
    prog_bar = ProgressMeter.Progress(n_run+1, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    @suppress while count < n_run+1
        tick();
        call(f, args...)
        t = tok()
        # println(t)
        time_log[count + 1] = t
        count += 1
        ProgressMeter.next!(prog_bar)
    end

    return time_log[2:end] # throw away the first run
end

"""
    noob_timing_sweep(f::Function, args_sweep; n_run = 100)

args_sweep: a tuple of different args for f to sweep over, eg., ars_sweep = (args1, args2, args3)
"""
function noob_timing_sweep(f, args_sweep; n_run = 100)
    time_log_sweep = zeros(n_run, length(args_sweep))

    for (i, args) in enumerate(args_sweep)
        time_log = noob_timing(f, args...; n_run = n_run)
        time_log_sweep[:, i] = time_log
    end
    return time_log_sweep
end



#####################
## plotting
#####################
# ribbon plot
function get_ribbons(time_logs)
    med_time = median(time_logs, dims=1)[:]
    # get 25th and 75th percentile
    q25s =  map(Base.Fix2(percentile, 25), eachcol(time_logs))
    q75s = map(Base.Fix2(percentile, 75), eachcol(time_logs))
    return med_time, q25s, q75s
end


function ribbon_plot(time_logs_cpu, time_logs_gpu, xs; xlabel = "dim of targets", ylabel = "time (s)", kwargs...)
    med_time, q25s, q75s = get_ribbons(time_logs_cpu)
    med_time_g, q25s_g, q75s_g = get_ribbons(time_logs_gpu)
    p = plot(xs, med_time, ribbon=(med_time-q25s, q75s-med_time), fillalpha=0.2, label="cpu")
    plot!(xs, med_time_g, ribbon=(med_time_g-q25s_g, q75s_g-med_time_g), fillalpha=0.2, label="gpu")
    plot!(xlabel=xlabel, ylabel=ylabel; kwargs...)
    return p
end

    

