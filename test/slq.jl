using Test
using Random
using LinearAlgebra
using SparseArrays
using MatrixDepot
using JLD2, FileIO
using Suppressor
using TraceEstimation

@suppress @testset "SLQ" begin
    Random.seed!(123543)
    @testset "Dense SPD Matrices" begin
        @testset "a[i,j] = exp(-2 * abS(i - j)) (small size)" begin
            println("Executing Test 01: Dense SPD - Small Size - Inverse")
            A = rand(610, 610)
            for i in 1:610
                for j in 1:610
                    A[i, j] = exp(-2 * abs(i - j))
                end
            end
            obv = slq(A, fn = inv, ctol = 0.01, eps = 0.05, mtol = 0.01)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=0.01)
        end
        @testset "a[i,j] = exp(-2 * abS(i - j)) (large size)" begin
            println("Executing Test 02: Dense SPD - Large Size - Inverse")
            A = rand(4610, 4610)
            for i in 1:4610
                for j in 1:4610
                    A[i, j] = exp(-2 * abs(i - j))
                end
            end
            obv = slq(A, fn = inv, ctol = 0.01, eps = 0.05, mtol = 0.01)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=0.01)
        end
        @testset "Random SPD Matrix (large size)" begin
            println("Executing Test 03: Random Dense SPD - Large Size - Log")
            A = rand(4000, 4000)
            A = A + A'
            while !isposdef(A)
                A = A + 30I
            end
            obv = slq(A, fn = log, ctol = 0.01)
            acv = tr(log(A))
            @test isapprox(obv, acv, rtol=0.01)
        end
        @testset "Hilbert Matrix" begin
            println("Executing Test 04: Dense Hilbert SPD - Inverse")
            A = matrixdepot("hilb", 3000)
            A = A + I
            obv = slq(A, fn = inv, ctol = 0.01)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=0.01)
        end
    end
    @testset "Sparse SPD Matrices" begin
        @testset "Sparse Matrix" begin
            println("Executing Test 05: Random Sparse Matrix - SQRT")
            A = sprand(6000, 6000, 0.001)
            A = A + A'
            while !isposdef(A)
                A = A + 60I
            end
            obv = slq(A, fn = sqrt, ctol = 0.01)
            M = Matrix(A)
            acv = tr(sqrt(M))
            @test isapprox(obv, acv, rtol = 0.01)
        end
        @testset "Wathen Sparse Matrix" begin
            println("Executing Test 06: Finite Element Matrix - Wathen")
            A = matrixdepot("wathen", 40) #7601x7601
            obv = slq(A, fn = inv, eps = 0.05)
            M = Matrix(A)
            acv = tr(inv(M))
            @test isapprox(obv, acv, rtol = 0.1)
        end
        @testset "TopOpt Shift +1 Matrix" begin
            println("Executing Test 07: TopOpt Shift +1")
            println("Loading from JLD2 file")
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
            obv = slq(K, fn = inv, eps = 0.005)
            M = Matrix(K)
            acv = tr(inv(M))
            @test isapprox(obv, acv, rtol=0.01)
        end
    end
end
