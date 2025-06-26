module InteractiveMPI

using Base.Threads: Threads, nthreads, @threads, SpinLock as Lock

include("julia/barrier.jl")

struct ThreadsPool
    size::Int
    barrier::Barrier
    lock::Lock
    channel::Channel{Nothing}
    sends::Vector{Any} # shared pool of send requests
end
ThreadsPool(n) = ThreadsPool(n, Barrier(n), Lock(), Channel{Nothing}(1), Any[])

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
(meth::MPI_Method)(args...; kwargs...) = meth.fun(meth.MPI, args...; kwargs...)

function start(main, nt)
    pool = ThreadsPool(nt)
    tasks = Task[]
    for rank in 1:nt
        MPI = ThreadsMPI(pool, rank)
        push!(tasks, Threads.@spawn main(MPI))
    end
    foreach(wait, tasks)
    return nothing
end

function Base.getproperty(MPI::ThreadsMPI, prop::Symbol)
    if hasfield(ThreadsMPI, prop)
        return getfield(MPI, prop)
    else
        meths = (; Init, Comm_size, Comm_rank, Barrier, Irecv!, Isend, Waitall)
        return MPI_Method(MPI, getproperty(meths, prop))
    end
end

Init(_) = nothing

Comm_size(::ThreadsMPI, comm::ThreadsCommunicator) = comm.pool.size
Comm_rank(::ThreadsMPI, comm::ThreadsCommunicator) = comm.rank - 1

Barrier(::ThreadsMPI, comm::ThreadsCommunicator) = wait(comm.pool.barrier)

struct Request{Msg}
    is_recv::Bool
    comm::ThreadsCommunicator
    source::Int
    dest::Int
    tag::Int
    msg::Msg
end

function Irecv!(::ThreadsMPI, msg, comm::ThreadsCommunicator; source, tag)
    return Request(true, comm, source, comm.rank - 1, tag, msg)
end
function Isend(::ThreadsMPI, msg, comm::ThreadsCommunicator; dest, tag)
    return Request(false, comm, comm.rank - 1, dest, tag, msg)
end

function Waitall(mpi::ThreadsMPI, reqs::Vector{<:Request})
    pool = mpi.COMM_WORLD.pool
    # collect send requests into pool.sends
    put!(pool.channel, nothing)
    for req in reqs
        if !req.is_recv
            push!(pool.sends, req)
        end
    end
    take!(pool.channel)
    # copy data from relevant send requests
    wait(pool.barrier)
    for req in reqs
        if req.is_recv
            for source in pool.sends
                if (req.source, req.dest) == (source.source, source.dest)
                    copy!(req.msg, source.msg)
                end
            end
        end
    end
    # cleanup after all threads have finished
    wait(pool.barrier) do 
        empty!(pool.sends)
    end

    return nothing
end

end # module
