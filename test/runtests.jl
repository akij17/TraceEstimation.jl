using SafeTestsets
using Test
using TraceEstimation

const METHOD = get(ENV, "METHOD", "all")

# SLQ
if METHOD in ("all", "slq")
    println("Running tests for SLQ")
    @safetestset "slq.jl" begin
        include("slq.jl")
    end
end

# cheby
if METHOD in ("all", "chebyhutch")
    println("Running tests for ChebyHutch")
    @safetestset "chebyhutch.jl" begin
        include("chebyhutch.jl")
    end
end

# diagonalapprox
if METHOD in ("all", "diagonalapprox")
    println("Running tests for DiagonalApprox")
    @safetestset "diagonalapprox.jl" begin
        include("diagonalapprox.jl")
    end
end

# Hutchinson
if METHOD in ("all", "hutchinson")
    println("Running tests for Hutchinson")
    @safetestset "hutchinson.jl" begin
        include("hutchinson.jl")
    end
end

#Diagonal Approximation
if METHOD in ("all", "diagapp")
    println("Running tests for diagapp")
    @safetestset "diagapp.jl" begin
        include("diagapp.jl")
    end
end
