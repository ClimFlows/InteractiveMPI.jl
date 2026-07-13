module InteractiveMPI

using Base.Threads: Threads, nthreads, @threads

# public: start, TrueMPIBackend, 

"""
    abstract type MPIBackend end
Parent type for MPI backends. Instances of such types are passed to `start`.
"""
abstract type MPIBackend end

"""
    InteractiveMPI.start(main, backend::MPIBackend, args...)

Invoke function `main` according to `backend`. `main` accepts as first argument an object `MPI` 
to be used as if it were the Julia `MPI` module. Extra arguments `args...` are passed to main after `MPI`.

    function main(MPI, args...)
        # use the object `MPI` as if it were the module `MPI`
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        ...
    end
"""
function start end

"""
    result = MPI.Critical(fun::Function)

Return `fun()` in a thread-safe way. 
Calls to thread-unsafe external libraries (like NetCDF) can be wrapped in a do-block:

    result = MPI.Critical() do
        ... # thread-unsafe code
    end

"""
function Critical end

# Pass-through MPI backend

"""
    backend = TrueMPIBackend()

Return `backend::MPIBackend` to be passed as second argument
to `start`. MPI invocations are then passed through to the MPI module. Use
this for production runs.
This backend is fully functional only if the Julia `MPI` module is loaded
by the main script or a project dependency:

    import MPI
    ...

"""
struct TrueMPIBackend <: MPIBackend end

@inline Base.getproperty(::TrueMPIBackend, sym::Symbol) = MPI_property(Val{sym}())
@inline MPI_property(::Val{:Critical}) = MPI_Critical
@inline MPI_Critical(fun) = fun()

# Threads-based MPI backend

include("julia/barrier.jl")

const Channels{T} = Vector{Channel{T}}
channels(T, n) = [Channel{T}(1) for _ in 1:n]

"""
    backend = ThreadsMPIBackend(size)

Return `backend` to be passed as second argument to `start`. 
This backend uses Julia threads instead of the MPI module. 
`start` will spawn `size` threads that act as `size` MPI processes.
"""
struct ThreadsMPIBackend <: MPIBackend
    size::Int
    barrier::Barrier
    channel::Channel{Nothing}
    data::Channels{Any}
    sends::Vector{Any} # shared pool of send requests
end
ThreadsMPIBackend(n::Int) = ThreadsMPIBackend(n, Barrier(n), Channel{Nothing}(1), channels(Any,n), Any[])

struct ThreadsCommunicator
    pool::ThreadsMPIBackend
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

function start(main, backend::ThreadsMPIBackend, args...)
    tasks = Task[]
    for rank in 1:backend.size
        MPI = ThreadsMPI(backend, rank-1)
        push!(tasks, Threads.@spawn main(MPI, args...))
    end
    foreach(wait, tasks)
    return nothing
end

@inline function Base.getproperty(MPI::ThreadsMPI, prop::Symbol)
    if hasfield(ThreadsMPI, prop)
        return getfield(MPI, prop)
    else
        meths = (; Critical, # needed for the threads-based implementation
                    Init, Comm_size, Comm_rank, Barrier, 
                    Irecv!, Isend, Waitall, # in point-to-point.jl
                     UBuffer, VBuffer, Scatter, Scatterv!, Gatherv!) # in scatter_gather.jl
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
