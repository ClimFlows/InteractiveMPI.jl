# borrowed from OhMyThreads.jl SimpleBarrier

mutable struct Barrier
    const n::Int64
    const c::Threads.Condition
    cnt::Int64

    function Barrier(n::Integer)
        new(n, Threads.Condition(), 0)
    end
end

Base.wait(b::Barrier) = wait(()->nothing, b)
function Base.wait(op, b::Barrier)
    lock(b.c)
    try
        b.cnt += 1
        if b.cnt == b.n # last arrived
            b.cnt = 0
            op() #user-defined cleanup operation performed once
            notify(b.c)
        else
            wait(b.c)
        end
    finally
        unlock(b.c)
    end
end
