using Base
using Test
using TraceEstimation

# SLQ
@testset "slq.jl" begin
    include("slq.jl")
end
@testset "chebyhutch.jl" begin
    include("chebyhutch.jl")
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
