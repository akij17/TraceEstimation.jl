using TraceEstimation
using Documenter

makedocs(;
    modules=[TraceEstimation],
    authors="Akshay Jain and contributors",
    repo="https://github.com/luca-aki/TraceEstimation.jl/blob/{commit}{path}#L{line}",
    sitename="TraceEstimation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://luca-aki.github.io/TraceEstimation.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/luca-aki/TraceEstimation.jl",
)
