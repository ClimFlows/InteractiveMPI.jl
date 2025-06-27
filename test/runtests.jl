using InteractiveMPI
using Test
using Base.Threads

using NetCDF: ncread
using ClimFlowsData: DYNAMICO_reader, DYNAMICO_meshfile
using CFDomains: VoronoiSphere

include("scatter.jl")

@testset "01-hello.jl" begin
    InteractiveMPI.start(40) do MPI
        MPI.Init()
        comm = MPI.COMM_WORLD
        for i in 1:4
            println("Hello world $i, I am rank $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))")
            MPI.Barrier(comm)
        end
    end
end

@testset "Critical" begin
    InteractiveMPI.start(2) do MPI
        MPI.Init()
        comm = MPI.COMM_WORLD
        sphere = MPI.Critical() do 
            meshfile = DYNAMICO_meshfile("uni.1deg.mesh.nc")
            VoronoiSphere(DYNAMICO_reader(ncread, meshfile) ; prec=Float32)
        end
    end
end

@testset "04-sendrecv.jl" begin
    InteractiveMPI.start(10) do MPI
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)

        dst = mod(rank+1, size)
        src = mod(rank-1, size)

        N = 4

        send_mesg = Array{Float64}(undef, N)
        recv_mesg = Array{Float64}(undef, N)

        fill!(send_mesg, Float64(rank))

        rreq = MPI.Irecv!(recv_mesg, comm; source=src, tag=src+32)

        print("$rank: Sending   $rank -> $dst = $send_mesg\n")
        sreq = MPI.Isend(send_mesg, comm; dest=dst, tag=rank+32)

        stats = MPI.Waitall([rreq, sreq])

        print("$rank: Received $src -> $rank = $recv_mesg\n")

        MPI.Barrier(comm)
    end
end

@testset "06-scatterv.jl" begin
    InteractiveMPI.start(test_scatter, 3)
end

