using InteractiveMPI
using Test

@testset "01-hello.jl" begin
    InteractiveMPI.start(10) do MPI
        MPI.Init()
        comm = MPI.COMM_WORLD
        print("Hello world, I am rank $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))\n")
        MPI.Barrier(comm)
        print("Hello world again, I am rank $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))\n")
    end
end

