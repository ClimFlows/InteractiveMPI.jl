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
