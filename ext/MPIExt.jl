module MPIExt

using MPI
import InteractiveMPI: MPI_property

@inline MPI_property(::Val{sym}) where sym = getproperty(MPI, sym)

end
