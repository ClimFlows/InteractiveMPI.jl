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
