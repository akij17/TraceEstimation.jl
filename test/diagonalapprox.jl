using Test
using Random
using LinearAlgebra
using SparseArrays
using MatrixDepot
using JLD2, FileIO
using MatrixMarket
using TraceEstimation

@testset "DiagonalApprox" begin
    Random.seed!(123543)
    @testset "sprandmatrices" begin
        println("Executing Test 01: sprand matrix")
        A = sprand(3000,3000,0.004)
        A = A + A' + 30I
        M = Matrix(A)
        act = tr(inv(M))
        obv = tr_inv(A, 4, 20)
        @test isapprox(act, obv, rtol = 0.2)
    end
    @testset "poissonmatrix" begin
        println("Executing Test 02: poisson matrix")
        A = matrixdepot("poisson", 50)
        M = Matrix(A)
        act = tr(inv(M))
        obv = tr_inv(A, 8, 40)
        @test isapprox(act, obv, rtol = 0.2)
    end
    @testset "wathenmatrix" begin
        println("Executing Test 03: wathen matrix")
        A = matrixdepot("wathen", 35)
        M = Matrix(A)
        act = tr(inv(M))
        obv = tr_inv(A, 8, 40, pc = "ilu")
        @test isapprox(act, obv, rtol = 0.2)
    end
    #=
    @testset "nasa2146" begin
        println("Executing Test 04: nasa2146 matrix")
        A = MatrixMarket.mmread("nasa2146.mtx")
        M = Matrix(A)
        act = tr(inv(M))
        obv = tr_inv(A, 8, 40)
        @test isapprox(act, obv, rtol = 0.2)
    end
    @testset "kuu" begin
        println("Executing Test 05: kuu matrix")
        A = MatrixMarket.mmread("Kuu.mtx")
        M = Matrix(A)
        act = tr(inv(M))
        obv = tr_inv(A, 8, 80)
        @test isapprox(act, obv, rtol = 0.2)
    end
    =#
    @testset "KMatrix" begin
        println("Executing Test 06: K matrix (high condition)")
        @load "topopt902.jld2" K
        K = SparseMatrixCSC(K)
        M = Matrix(K)
        act = tr(inv(M))
        obv = tr_inv(K, 8, 150)
        @test isapprox(act, obv, rtol = 0.2)
    end
end
