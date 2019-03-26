using LinearAlgebra
using SparseArrays
using TraceEstimation
using Random
using Test

function percent_error(obv::Float64, acv::Float64)
    e = (abs(obv-acv) / acv) * 100
    if(e < 10)
        return true
    else
        return false
    end
end

# Most of these tests will focus on Symmetric Hermitian Positive Definite Matrices
# as Hutchinson method works best for those in its original non-hybrid form for
# other kind of matrices and/or better accuracy Hybrid Hutchinson method or some
# other method should be used

@testset "Hutchinson" begin
    Random.seed!(1234323)
    @testset "Dense SPD Hermitian Matrices" begin
        @testset "a[i,j] = exp(-2 * abS(i - j)) (small size)" begin
            A = rand(610, 610)
            for i in 1:610
                for j in 1:610
                    A[i, j] = exp(-2 * abs(i - j))
                end
            end
            w = HutchWorkspace(A, N = 20, skipverify = true)
            @time obv = hutch!(w)
            @time acv = tr(inv(A))
            @test percent_error(obv, acv)
        end
        @testset "a[i,j] = exp(-2 * abS(i - j)) (large size)" begin
            A = rand(8100, 8100)
            for i in 1:8100
                for j in 1:8100
                    A[i, j] = exp(-2 * abs(i - j))
                end
            end
            w = HutchWorkspace(A, N = 20, skipverify = true)
            @time obv = hutch!(w)
            @time acv = tr(inv(A))
            @test percent_error(obv, acv)
        end
        @testset "Random generated SPD matrix (small size) (N=30)" begin
            A = rand(810, 810)
            A = A + A' + 30I
            while isposdef(A) == false
                A = rand(810, 810)
                A = A + A' + 30I
            end
            w = HutchWorkspace(A, N = 30, skipverify = true)
            @time obv = hutch!(w)
            @time acv = tr(inv(A))
            @test percent_error(obv, acv)
        end
        @testset "Random generated SPD matrix (small size) (N=60)" begin
            A = rand(810, 810)
            A = A + A' + 30I
            while isposdef(A) == false
                A = rand(810, 810)
                A = A + A' + 30I
            end
            w = HutchWorkspace(A, N = 60, skipverify = true)
            @time obv = hutch!(w)
            @time acv = tr(inv(A))
            @test percent_error(obv, acv)
        end
        @testset "Random generated SPD matrix (large size) (N=30)" begin
            A = rand(8100, 8100)
            A = A + A' + 300I
            while isposdef(A) == false
                A = rand(8100, 8100)
                A = A + A' + 300I
            end
            w = HutchWorkspace(A, N = 30, skipverify = true)
            @time obv = hutch!(w)
            @time acv = tr(inv(A))
            @test percent_error(obv, acv)
        end
    end
    @testset "Sparse SPD Hermitian Matrices" begin
        @testset "Random generated Sparse SPD (small size)"
            A = Symmetric(sprand(1000, 1000, 0.7))
            A = A+50*I
            while isposdef(A) == false
                A = Symmetric(sprand(1000, 1000, 0.7))
                A = A+50*I
            end
            w = HutchWorkspace(A, N = 30, skipverify = true)
            @time obv = hutch!(w)
            @time acv = tr(inv(A))
            @test percent_error(obv, acv)
        end
    end
end
