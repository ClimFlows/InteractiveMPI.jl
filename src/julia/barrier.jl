struct Barrier
    count::Int
    arrived::Threads.Atomic{Int}
    release::Channel{Nothing}
end

function Barrier(n::Int)
    Barrier(n, Threads.Atomic{Int}(0), Channel{Nothing}(n))
end

function wait_at_barrier(b::Barrier)
    if Threads.atomic_add!(b.arrived, 1) == b.count-1 
        for _ in 1:b.count
            put!(b.release, nothing)
        end
        b.arrived[] = 0
    end
    take!(b.release)
end
