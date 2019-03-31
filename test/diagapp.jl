using Test
using Random
using LinearAlgebra
using SparseArrays
using JLD2, FileIO
using TraceEstimation

# Diagonal Approximation method as described by P. Fika works only for SPD matrix with low condition number (< 500)
@testset "Diagonal Approximation (P.Fika)" begin
    Random.seed!(1234323)
    @testset "Dense SPD Hermitian Matrices" begin
        @testset "a[i,j] = exp(-2 * abS(i - j)) (small size)" begin
            println("Executing Test 01: Dense SPD Small Size")
            A = rand(610, 610)
            for i in 1:610
                for j in 1:610
                    A[i, j] = exp(-2 * abs(i - j))
                end
            end
            obv = diagapp(A)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=10)
        end
        @testset "a[i,j] = exp(-2 * abS(i - j)) (large size)" begin
            println("Executing Test 02: Dense SPD Large Size")
            A = rand(8100, 8100)
            for i in 1:8100
                for j in 1:8100
                    A[i, j] = exp(-2 * abs(i - j))
                end
            end
            obv = diagapp(A)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=10)
        end
        @testset "Random generated SPD matrix (small size) (N=30)" begin
            println("Executing Test 03: Random SPD Small Size (N = 30)")
            A = rand(810, 810)
            A = A + A' + 30I
            while isposdef(A) == false
                A = rand(810, 810)
                A = A + A' + 30I
            end
            obv = diagapp(A)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=10)
        end
        @testset "Random generated SPD matrix (small size) (N=60)" begin
            println("Executing Test 04: Random SPD Small Size (N=60)")
            A = rand(810, 810)
            A = A + A' + 30I
            while isposdef(A) == false
                A = rand(810, 810)
                A = A + A' + 30I
            end
            obv = diagapp(A)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=10)
        end
        @testset "Random generated SPD matrix (large size) (N=30)" begin
            println("Executing Test 05: Random SPD Large Size")
            A = rand(8100, 8100)
            A = A + A' + 300I
            while isposdef(A) == false
                A = rand(8100, 8100)
                A = A + A' + 300I
            end
            obv = diagapp(A)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=10)
        end
    end
    @testset "Random generated SPD matrix (large size, large condition number)" begin
            println("Executing Test 05.5: Random SPD Large Size Larger Condition")
            A = Symmetric(rand(5005, 5005))
            A = A + 10I
            while isposdef(A) == false
                A = A + 10I
            end
            obv = diagapp(A)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=10)
        end
    @testset "Sparse SPD Hermitian Matrices" begin
        @testset "Random generated Sparse SPD (small size)" begin
            println("Executing Test 06: Sparse Random SPD Small Size")
            A = Symmetric(sprand(1000, 1000, 0.7))
            A = A+50*I
            while isposdef(A) == false
                A = Symmetric(sprand(1000, 1000, 0.7))
                A = A+50*I
            end
            obv = diagapp(A)
            A = Matrix(A)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=10)
        end
        #= Condition number too large (> 1000)
        @testset "TopOpt Test (Large Condition Number)" begin
            println("Executing Test 07: TopOpt Large Condition Number")
            s = (40, 10) # increase to increase the matrix size
            xmin = 0.9 # decrease to increase the condition number
            problem = HalfMBB(Val{:Linear}, s, (1.0, 1.0), 1.0, 0.3, 1.0)
            solver = FEASolver(Displacement, Direct, problem, xmin = xmin)
            n = length(solver.vars)
            solver.vars[rand(1:n, n÷2)] .= 0
            solver()
            K = solver.globalinfo.K
            obv = diagapp(K)
            M = Matrix(K)
            acv = tr(inv(M))
            @test isapprox(obv, acv, rtol=10)
        end
        =#
        @testset "TopOpt Test (Modified Condition Number)" begin
            println("Executing Test 08: TopOpt Modified Condition Number")
            println("Loading JLD2 file")
            #=
            s = (40, 10) # increase to increase the matrix size
            xmin = 0.9 # decrease to increase the condition number
            problem = HalfMBB(Val{:Linear}, s, (1.0, 1.0), 1.0, 0.3, 1.0)
            solver = FEASolver(Displacement, Direct, problem, xmin = xmin)
            n = length(solver.vars)
            solver.vars[rand(1:n, n÷2)] .= 0
            solver()
            K = solver.globalinfo.K
            K = K+1*I
            =#
            @load "topoptfile.jld2" K
            obv = diagapp(K)
            M = Matrix(K)
            acv = tr(inv(M))
            @test isapprox(obv, acv, rtol=10)
        end
    end
    #=
    @testset "CuArrays Test" begin
        @testset "Dense Random CuArray" begin
            println("Executing Test 09: CuArray Small Size")
            A = cu(rand(400,400))
            A = A+A'+40*I
            function mycurand(type::Type, size::Int64)
                cu(rand(type, size))
            end
            function mycurand(range::StepRange{<:Any, <:Any}, size::Int64)
                cu(rand(range, size))
            end            
            obv = diagapp(A, randfunc=mycurand)
            M = Matrix(A)
            acv = tr(inv(M))
            @test isapprox(obv, acv, rtol=10)
        end
    end =#
end

