using TraceEstimation
using Documenter

makedocs(;
    modules=[TraceEstimation],
    authors="Mohamed Tarek <mohamed82008@gmail.com> and contributors",
    repo="https://github.com/mohamed82008/TraceEstimation.jl/blob/{commit}{path}#L{line}",
    sitename="TraceEstimation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mohamed82008.github.io/TraceEstimation.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mohamed82008/TraceEstimation.jl",
)
