# examples/06-scatterv.jl
# This example shows how to use MPI.Scatterv! and MPI.Gatherv!
# roughly based on the example from
# https://stackoverflow.com/a/36082684/392585

"""
    split_count(N::Integer, n::Integer)

Return a vector of `n` integers which are approximately equally sized and sum to `N`.
"""
function split_count(N::Integer, n::Integer)
    q, r = divrem(N, n)
    return [i <= r ? q + 1 : q for i in 1:n]
end

const MPI_root = 0

function test_scatter(MPI)
    MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    if rank == MPI_root
        M, N = 4, 7

        test = Float64[i+j for i in 1:M, j in 1:N]
        output = similar(test)

        # Julia arrays are stored in column-major order, so we need to split along the last dimension
        M_counts = [M for i in 1:comm_size]
        N_counts = split_count(N, comm_size)

        # store sizes in 2 * comm_size Array
        sizes = vcat(M_counts', N_counts')
        size_ubuf = MPI.UBuffer(sizes, 2)

        # store number of values to send to each rank in comm_size length Vector
        counts = vec(prod(sizes; dims=1))

        test_vbuf = MPI.VBuffer(test, counts) # VBuffer for scatter
        output_vbuf = MPI.VBuffer(output, counts) # VBuffer for gather
    else
        # these variables can be set to `nothing` on non-root processes
        size_ubuf = MPI.UBuffer(nothing)
        output_vbuf = test_vbuf = MPI.VBuffer(nothing)
    end

    if rank == MPI_root
        println("Original matrix")
        println("================")
        @show test sizes counts
        println()
        println("Each rank")
        println("================")
    end

    MPI.Barrier(comm)

    local_size = MPI.Scatter(size_ubuf, NTuple{2,Int}, MPI_root, comm)
    local_test = MPI.Scatterv!(test_vbuf, zeros(Float64, local_size), MPI_root, comm)

    for i in 0:(comm_size - 1)
        if rank == i
             @show rank local_test
        end
        MPI.Barrier(comm)
    end

    MPI.Gatherv!(local_test, output_vbuf, MPI_root, comm)

    if rank == MPI_root
        println()
        println("Final matrix")
        println("================")
        @show output
    end
end
