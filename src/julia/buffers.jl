struct TUBuffer{A}
    data::A
    count::Int
    nchunks::Union{Nothing,Int}
end
function TUBuffer(arr::AbstractArray, count::Int)
    @assert stride(arr, 1) == 1
    return TUBuffer(arr, count, div(length(arr), count))
end
TUBuffer(::Nothing) = TUBuffer(nothing, 0, nothing)

Base.similar(buf::TUBuffer) = TUBuffer(similar(buf.data), buf.count, buf.nchunks)

struct TVBuffer{A}
    data::A
    counts::Vector{Int}
    displs::Vector{Int}
end

function TVBuffer(arr::AbstractArray, counts::Vector{Int})
    @assert stride(arr,1) == 1
    displs = similar(counts)
    d = zero(Int)
    for i in eachindex(displs)
        displs[i] = d
        d += counts[i]
    end
    @assert length(arr) >= d
    TVBuffer(arr, counts, displs)
end

TVBuffer(::Nothing) = TVBuffer(nothing, Int[], Int[])
