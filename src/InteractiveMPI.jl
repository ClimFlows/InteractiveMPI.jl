module InteractiveMPI

    using Base.Threads: Threads, nthreads, @threads

    include("julia/barrier.jl")

    struct ThreadsPool
        size::Int
        barrier::Barrier
    end
    ThreadsPool(n) = ThreadsPool(n, Barrier(n))

    struct ThreadsCommunicator
        pool::ThreadsPool
        rank::Int
    end

    struct ThreadsMPI
        COMM_WORLD::ThreadsCommunicator
    end
    ThreadsMPI(pool, rank) = ThreadsMPI(ThreadsCommunicator(pool, rank))

    struct MPI_Method{Fun}
        MPI::ThreadsMPI
        fun::Fun
    end
    (meth::MPI_Method)(args...) = meth.fun(meth.MPI, args...)

    function start(main, nt)
        pool = ThreadsPool(nt)
        tasks = []
        for rank in 1:nt
            MPI = ThreadsMPI(pool, rank)
            push!(tasks, Threads.@spawn main(MPI))
        end
        foreach(wait, tasks)
    end

    function Base.getproperty(MPI::ThreadsMPI, prop::Symbol)
        if hasfield(ThreadsMPI, prop)
            return getfield(MPI, prop)
        else
            meths = (; Init, Comm_size, Comm_rank, Barrier)
            return MPI_Method(MPI, getproperty(meths, prop))
        end
    end

    Init(_) = nothing

    Comm_size(::ThreadsMPI, comm::ThreadsCommunicator) = comm.pool.size 
    Comm_rank(::ThreadsMPI, comm::ThreadsCommunicator) = comm.rank-1 

    Barrier(::ThreadsMPI, comm::ThreadsCommunicator) = wait_at_barrier(comm.pool.barrier)

end
