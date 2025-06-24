using InteractiveMPI
using Documenter

DocMeta.setdocmeta!(InteractiveMPI, :DocTestSetup, :(using InteractiveMPI); recursive=true)

makedocs(;
    modules=[InteractiveMPI],
    authors="Thomas Dubos <thomas.dubos@polytechnique.edu> and contributors",
    sitename="InteractiveMPI.jl",
    format=Documenter.HTML(;
        canonical="https://dubosipsl.github.io/InteractiveMPI.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dubosipsl/InteractiveMPI.jl",
    devbranch="main",
)
