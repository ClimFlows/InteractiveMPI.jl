module InteractiveMPI

using Base.Threads: Threads, nthreads, @threads, SpinLock as Lock

include("julia/barrier.jl")
include("julia/buffers.jl")

struct ThreadsPool
    size::Int
    barrier::Barrier
    lock::Lock
    channel::Channel{Nothing}
    data::Channel{Any}
    sends::Vector{Any} # shared pool of send requests
end
ThreadsPool(n) = ThreadsPool(n, Barrier(n), Lock(), Channel{Nothing}(1), Channel{Any}(n), Any[])

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
        meths = (; Init, Comm_size, Comm_rank, Barrier, Irecv!, Isend, Waitall,
                 Critical, UBuffer, VBuffer, Scatter, Scatterv!)
        return MPI_Method(MPI, getproperty(meths, prop))
    end
end

Init(_) = nothing

Comm_size(::ThreadsMPI, comm::ThreadsCommunicator) = comm.pool.size
Comm_rank(::ThreadsMPI, comm::ThreadsCommunicator) = comm.rank

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
    return Request(true, comm, source, comm.rank, tag, msg)
end
function Isend(::ThreadsMPI, msg, comm::ThreadsCommunicator; dest, tag)
    return Request(false, comm, comm.rank, dest, tag, msg)
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
        return empty!(pool.sends)
    end

    return nothing
end

function Critical(MPI::ThreadsMPI, todo::F) where {F}
    channel = MPI.COMM_WORLD.pool.channel
    put!(channel, nothing)
    result = todo()
    take!(channel)
    return result
end

UBuffer(::ThreadsMPI, args...) = TUBuffer(args...)

VBuffer(::ThreadsMPI, args...) = TVBuffer(args...)

function Scatter(MPI::ThreadsMPI, buf::TUBuffer, ::Type{T}, origin, comm) where T
    @info "Scatter" origin comm.rank
    pool = comm.pool
    if comm.rank == origin
        for i in 1:pool.size # share data
            put!(pool.data, (buf.data, buf.count))
        end
    end
    (data, count) = take!(pool.data)
    return TScatter(data, comm.rank, count, T)
end

function TScatter(data::AbstractArray, rank, count, ::Type{T}) where T
    start, stop = rank*count, (rank+1)*count-1
    result = T(data[(begin+start):(begin+stop)])
    @info "TScatter" data rank count result
    return result
end

function Scatterv!(MPI::ThreadsMPI, buf::TVBuffer, output, origin, comm)
    @info "Scatterv!" buf origin comm.rank
    pool = comm.pool
    if comm.rank == origin
        for i in 1:pool.size # share data
            put!(pool.data, (buf.data, buf.counts, buf.displs))
        end
    end
    (data, counts, displs) = take!(pool.data)
    return TScatterv!(data, output, comm.rank, counts, displs)
end

function TScatterv!(data::AbstractArray, output::AbstractArray, rank, counts, displs)
    start, stop = displs[rank+1], displs[rank+1]+counts[rank+1]-1
    @info "TScatterv!" rank start stop length(data) length(output)
    copyto!(output, @view data[(begin+start):(begin+stop)])
    return output
end

end # module
