using Test
using Random
using LinearAlgebra
using SparseArrays
using MatrixDepot
using JLD2, FileIO
using TraceEstimation

function MAPE(A::Vector, O::Vector)
    return 1/size(A, 1) * sum(abs.((A .- O) ./ A))
end

@testset "Cheby-Hutch" begin
    Random.seed!(123432)
    @testset "Dense Matrices" begin
        @testset "a[i,j] = exp(-2 * abS(i - j)) (small size)" begin
            println("Executing Test 01: Dense SPD - Small Size - Inverse")
            A = rand(610, 610)
            for i in 1:610
                for j in 1:610
                    A[i, j] = exp(-2 * abs(i - j))
                end
            end
            obv = chebyhutch(A, 4, 6, fn = inv)
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
            obv = chebyhutch(A, 4, 6, fn = inv)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=0.01)
        end
        @testset "Random SPD Matrix (large size)" begin
            println("Executing Test 03: Random Dense SPD - Large Size - Log Determinant")
            A = rand(4610, 4610)
            A = A + A'
            while !isposdef(A)
                A = A + 30I
            end
            obv = chebyhutch(A, 4, 8, fn = log)
            acv = logdet(A)
            @test isapprox(obv, acv, rtol=0.02)
        end
        @testset "Hilbert Matrix" begin
            println("Executing Test 04: Dense Hilbert SPD - Inverse")
            A = matrixdepot("hilb", 3000)
            A = A + I
            obv = chebyhutch(A, 4, 6, fn = inv)
            acv = tr(inv(A))
            @test isapprox(obv, acv, rtol=0.01)
        end
    end
    @testset "Sparse Matrices" begin
        @testset "Random Sparse Matrix" begin
            println("Executing Test 05: Random Sparse Matrix - SQRT")
            A = sprand(6000, 6000, 0.001)
            A = A + A'
            while !isposdef(A)
                A = A + 60I
            end
            obv = chebyhutch(A, 4, 6, fn = sqrt)
            M = Matrix(A)
            acv = tr(sqrt(M))
            @test isapprox(obv, acv, rtol = 0.01)
        end
        @testset "Wathen Sparse Matrix" begin
            println("Executing Test 06: Finite Element Matrix - Wathen")
            A = matrixdepot("wathen", 40) #7601x7601
            obv = chebyhutch(A, 6, 12, fn = inv)
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
            obv = chebyhutch(K, 4, 8, fn = inv)
            M = Matrix(K)
            acv = tr(inv(M))
            @test isapprox(obv, acv, rtol=0.01)
        end
    end
end
@testset "Cheby-Diagonal" begin
    @testset "Dense Matrices" begin
        @testset "Random SPD Matrix (large size)" begin
            println("Executing Test 2.1: Random Dense SPD - Large Size - Inverse")
            A = rand(4610, 4610)
            A = A + A'
            while !isposdef(A)
                A = A + 30I
            end
            o = chebydiagonal(A, 4, 8, fn = inv)
            a = diag(inv(A))
            @test MAPE(a, o) <= 0.2
        end
        @testset "Hilbert Matrix" begin
            println("Executing Test 2.2: Dense Hilbert SPD - Inverse")
            A = matrixdepot("hilb", 3000)
            A = A + I
            o = chebydiagonal(A, 4, 22, fn = inv)
            a = diag(inv(A))
            @test MAPE(a, o) <= 0.2
        end
    end
    @testset "Sparse Matrices" begin
        @testset "Random Sparse Matrix" begin
            println("Executing Test 2.3: Random Sparse Matrix - SQRT")
            A = sprand(6000, 6000, 0.001)
            A = A + A'
            while !isposdef(A)
                A = A + 60I
            end
            o = chebydiagonal(A, 4, 6, fn = sqrt)
            M = Matrix(A)
            a = diag(sqrt(M))
            @test MAPE(a, o) <= 0.2
        end
        @testset "Wathen Sparse Matrix" begin
            println("Executing Test 2.4: Finite Element Matrix - Wathen")
            A = matrixdepot("wathen", 40) #7601x7601
            o = chebydiagonal(A, 4, 22, fn = log)
            M = Matrix(A)
            a = diag(log(M))
            @test MAPE(a, o) <= 0.2
        end
    end
end
