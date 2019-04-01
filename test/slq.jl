using Test
using Random
using LinearAlgebra
using SparseArrays
using JLD2, FileIO
using TraceEstimation

# Most of these tests will focus on Symmetric Positive Definite Matrices with low condition number (< 500)
# as SLQ works best for SPD matrices with low condition number
# A hybrid for larger condition and semi-positive definite will also be availabe

@testset "SLQ" begin
    Random.seed!(1234323)
    @testset "Dense SPD Matrices" begin
        @testset "a[i,j] = exp(-2 * abS(i - j)) (small size, invfun)" begin
            println("Executing Test 01: Dense SPD Small Size (inverse function)")
            A = rand(610, 610)
            for i in 1:610
                for j in 1:610
                    A[i, j] = exp(-2 * abs(i - j))
                end
            end
            w = SLQWorkspace(A, m = 20, ctol = 0.2)
            obv = slq(w)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=10)
        end
        @testset "a[i,j] = exp(-2 * abS(i - j)) (small size, sqrfun)" begin
            println("Executing Test 01: Dense SPD Small Size (square function)")
            A = rand(610, 610)
            for i in 1:610
                for j in 1:610
                    A[i, j] = exp(-2 * abs(i - j))
                end
            end
            sqrfun(x) = x^2
            w = SLQWorkspace(A, f = sqrfun, m = 20, ctol = 0.2)
            obv = slq(w)
            acv = tr(A^2)
            @test isapprox(obv, acv, rtol=10)
        end
        @testset "a[i,j] = exp(-2 * abS(i - j)) (large size, invfun)" begin
            println("Executing Test 02: Dense SPD Large Size (inverse function)")
            A = rand(5100, 5100)
            for i in 1:5100
                for j in 1:5100
                    A[i, j] = exp(-2 * abs(i - j))
                end
            end
            w = SLQWorkspace(A)
            obv = slq(w, skipverify = true)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=10)
        end
        @testset "a[i,j] = exp(-2 * abS(i - j)) (large size, invfun)" begin
            println("Executing Test 02: Dense SPD Large Size (square function)")
            A = rand(5100, 5100)
            for i in 1:5100
                for j in 1:5100
                    A[i, j] = exp(-2 * abs(i - j))
                end
            end
            w = SLQWorkspace(A, f = sqrfun)
            obv = slq(w, skipverify = true)
            acv = tr(A^2)
            @test isapprox(obv, acv, rtol=10)
        end
        @testset "Random generated SPD matrix (small size, invfun)" begin
            println("Executing Test 03: Random SPD Small Size (inverse function)")
            A = rand(810, 810)
            A = A + A' + 30I
            while isposdef(A) == false
                A = rand(810, 810)
                A = A + A' + 30I
            end
            w = SLQWorkspace(A)
            obv = slq(w, skipverify = true)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=10)
        end
        @testset "Random generated SPD matrix (small size)" begin
            println("Executing Test 04: Random SPD Small Size (square function)")
            A = rand(810, 810)
            A = A + A' + 30I
            while isposdef(A) == false
                A = rand(810, 810)
                A = A + A' + 30I
            end
            w = SLQWorkspace(A, f = sqrfun)
            obv = slq(w, skipverify = true)
            acv = tr(A^2)
            @test isapprox(obv, acv, rtol=10)
        end
        @testset "Random generated SPD matrix (large size)" begin
            println("Executing Test 05: Random SPD Large Size (inverse function)")
            A = rand(5005, 5005)
            A = A + A' + 300I
            while isposdef(A) == false
                A = rand(5005, 5005)
                A = A + A' + 300I
            end
            w = SLQWorkspace(A)
            obv = slq(w, skipverify = true)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=10)
        end
        @testset "Random generated SPD matrix (large size, large condition number)" begin
            println("Executing Test 05.5: Random SPD Large Size Larger Condition (inverse function)")
            A = Symmetric(rand(5005, 5005))
            A = A + 10I
            while isposdef(A) == false
                A = A + 10I
            end
            w = SLQWorkspace(A)
            obv = slq(w, skipverify = true)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=10)
        end
    end
    @testset "Sparse SPD Hermitian Matrices" begin
        @testset "Random generated Sparse SPD (small size)" begin
            println("Executing Test 06: Sparse Random SPD Small Size (inverse function)")
            A = Symmetric(sprand(1000, 1000, 0.7))
            A = A+50*I
            while isposdef(A) == false
                A = Symmetric(sprand(1000, 1000, 0.7))
                A = A+50*I
            end
            w = SLQWorkspace(A)
            obv = slq(w, skipverify = true)
            A = Matrix(A)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=10)
        end
        #= Condtion Number too large ( > 1000)
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
            w = HutchWorkspace(K)
            obv = hutch!(w)
            M = Matrix(K)
            acv = tr(inv(M))
            @test isapprox(obv, acv, rtol=10)
        end
        =#
        @testset "TopOpt Test (Modified Condition Number)" begin
            println("Executing Test 08: TopOpt Modified Condition Number (inverse function)")
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
            w = SLQWorkspace(K)
            obv = slq(w, skipverify = true)
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
            f(n) = cu(rand(-1.0:2.0:1.0, n))
            w = HutchWorkspace(A, f)
            obv = hutch!(w)
            M = Matrix(A)
            acv = tr(inv(M))
            @test isapprox(obv, acv, rtol=10)
        end
    end =#
end
