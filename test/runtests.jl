using Base
using Test
using TraceEstimation

# Hutchinson
@testset "hutchinson.jl" begin
    include("hutchinson.jl")
end

#Diagonal Approximation
@testset "diagapp.jl" begin
    include("diagapp.jl")
end

# SLQ
@testset "slq.jl" begin
    include("slq.jl")
end
