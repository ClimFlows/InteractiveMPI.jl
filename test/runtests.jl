using InteractiveMPI
using Test
using Base.Threads

#=
function start_test_barrier(nt)
    @sync begin
        b = InteractiveMPI.Barrier(nt)
        for rank = 1:nt
            @spawn     for i in 1:4
                println("Task $rank at step $i")
                wait(b)
            end
        end
    end
end

@testset "Barrier" start_test_barrier(40)
=#

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
