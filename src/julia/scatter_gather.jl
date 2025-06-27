UBuffer(::ThreadsMPI, args...) = TUBuffer(args...)

VBuffer(::ThreadsMPI, args...) = TVBuffer(args...)

function Scatter(MPI::ThreadsMPI, buf::TUBuffer, ::Type{T}, sender, comm) where T
    @info "Scatter" sender comm.rank
    (; pool, rank) = comm
    if rank == sender
        for i in 1:pool.size # share data
            start, stop = (i-1)*buf.count, i*buf.count-1
            put!(pool.data[i], @view buf.data[begin+start:begin+stop])
        end
    end
    return T(take!(pool.data[rank+1]))    
end

function Scatterv!(MPI::ThreadsMPI, buf::TVBuffer, output, sender, comm)
    @info "Scatterv!" buf sender comm.rank
    (; pool, rank) = comm
    if rank == sender
        (; counts, displs) = buf
        for i in 1:pool.size # share data
            start, stop = displs[i], displs[i]+counts[i]-1
            put!(pool.data[i], @view buf.data[begin+start:begin+stop])
        end
    end
    return copyto!(output, take!(pool.data[rank+1]))
end
