module InteractiveMPI

using Base.Threads: Threads, nthreads, @threads

include("julia/barrier.jl")

struct ThreadsPool
    size::Int
    barrier::Barrier
    channel::Channel{Nothing}
    data::Channel{Any}
    sends::Vector{Any} # shared pool of send requests
end
ThreadsPool(n) = ThreadsPool(n, Barrier(n), Channel{Nothing}(1), Channel{Any}(n), Any[])

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
        MPI = ThreadsMPI(pool, rank-1)
        push!(tasks, Threads.@spawn main(MPI))
    end
    foreach(wait, tasks)
    return nothing
end

function Base.getproperty(MPI::ThreadsMPI, prop::Symbol)
    if hasfield(ThreadsMPI, prop)
        return getfield(MPI, prop)
    else
        meths = (; Critical, # needed for the threads-based implementation
                    Init, Comm_size, Comm_rank, Barrier, 
                    Irecv!, Isend, Waitall, # in point-to-point.jl
                     UBuffer, VBuffer, Scatter, Scatterv!) # in scatter_gather.jl
        return MPI_Method(MPI, getproperty(meths, prop))
    end
end

function Critical(MPI::ThreadsMPI, todo::F) where {F}
    channel = MPI.COMM_WORLD.pool.channel
    put!(channel, nothing)
    result = todo()
    take!(channel)
    return result
end

Init(_) = nothing

Comm_size(::ThreadsMPI, comm::ThreadsCommunicator) = comm.pool.size
Comm_rank(::ThreadsMPI, comm::ThreadsCommunicator) = comm.rank

Barrier(::ThreadsMPI, comm::ThreadsCommunicator) = wait(comm.pool.barrier)

include("julia/point-to-point.jl")
include("julia/buffers.jl")
include("julia/scatter_gather.jl")

end # module
