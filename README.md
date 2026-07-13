# InteractiveMPI

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ClimFlows.github.io/InteractiveMPI.jl/dev/)
[![Build Status](https://github.com/ClimFlows/InteractiveMPI.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ClimFlows/InteractiveMPI.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ClimFlows/InteractiveMPI.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ClimFlows/InteractiveMPI.jl)

InteractiveMPI is a Julia package that provides an interface for writing MPI (Message Passing Interface) programs that can run both with real MPI and with Julia threads for easier debugging and testing.

## Overview

This package enables developers to write MPI applications that can be executed in two modes:
1. **Real MPI mode**: Uses actual MPI calls through the Julia MPI module
2. **Thread-based mode**: Simulates MPI behavior using Julia threads, allowing for easier debugging and testing without requiring an MPI environment

## Key Features

- **Dual Backend Support**: Switch between real MPI and thread-based simulation seamlessly
- **Thread-Safe Operations**: Provides thread-safe wrappers for external libraries
- **Full MPI Functionality**: Implements core MPI operations including point-to-point communication, scatter/gather operations, and barriers
- **Easy Testing**: Run MPI code locally with Julia threads without needing MPI installation

## Installation

Add InteractiveMPI to your Julia environment:

```julia
using Pkg
Pkg.add("InteractiveMPI")
```

To use the real MPI backend, you'll also need to install the MPI.jl package:

```julia
Pkg.add("MPI")
```

## Usage

### Basic Usage Pattern

```julia
using InteractiveMPI

# Define your main function that takes an MPI object
function main_function(MPI, args...)
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    # Your MPI code here
    println("Hello from rank $rank of $size")
    
    MPI.Barrier(comm)
end

# Run with thread-based backend (for testing/debugging)
InteractiveMPI.start(main_function, InteractiveMPI.ThreadsMPIBackend(4))

# Run with real MPI backend (for production)
InteractiveMPI.start(main_function, InteractiveMPI.TrueMPIBackend())
```

### Thread-Based Backend

The `ThreadsMPIBackend` creates Julia threads that simulate MPI processes:

```julia
# Run with 4 simulated MPI processes
InteractiveMPI.start(main_function, InteractiveMPI.ThreadsMPIBackend(4))
```

### Real MPI Backend

The `TrueMPIBackend` passes MPI calls through to the actual MPI implementation:

```julia
# Requires MPI.jl to be loaded
using MPI
InteractiveMPI.start(main_function, InteractiveMPI.TrueMPIBackend())
```

### Thread-Safe Critical Sections

For thread-unsafe operations (like those involving NetCDF), wrap them in `MPI.Critical`:

```julia
result = MPI.Critical() do
    # Thread-unsafe code here
    # e.g., reading from NetCDF files
end
```

## Supported MPI Operations

The package implements the following MPI operations:

- Point-to-point communication: `Irecv!`, `Isend`, `Waitall`
- Collective operations: `Barrier`, `Comm_size`, `Comm_rank`
- Scatter/Gather operations: `Scatter`, `Scatterv!`, `Gatherv!`
- Buffer management: `UBuffer`, `VBuffer`

## Testing

The package includes comprehensive tests that demonstrate usage patterns:

```julia
# Run tests
using InteractiveMPI
include("test/runtests.jl")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.