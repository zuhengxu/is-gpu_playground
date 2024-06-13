using CUDA
using StatsFuns

function my_kernel(f, in, out)
    i = threadIdx().x
    out[i] = f(in[i])
    return
end

function main()
    size = 1000
    in = cu(randn(size))
    out = CuArray{Float32}(undef, size)
    @cuda threads=size my_kernel(normpdf, in , out)
    @show out
end

main()