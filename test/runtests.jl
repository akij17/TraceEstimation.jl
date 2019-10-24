using Base
using Test
using TraceEstimation

# SLQ
println("Running tests for SLQ")
@testset "slq.jl" begin
    include("slq.jl")
end
# cheby
println("Running tests for ChebyHutch")
@testset "chebyhutch.jl" begin
    include("chebyhutch.jl")
end

# diagonalapprox
println("Running tests for DiagonalApprox")
@testset "diagonalapprox.jl" begin
    include("diagonalapprox.jl")
end



#=
# Hutchinson
@testset "hutchinson.jl" begin
    include("hutchinson.jl")
end

#Diagonal Approximation
@testset "diagapp.jl" begin
    include("diagapp.jl")
end
=#
