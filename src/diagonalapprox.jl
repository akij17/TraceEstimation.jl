# Diagonal Approximation algorithm

# step 1: compute M = diag(Z^-1)
# step 2: compute fitting sample Sfit
using LinearAlgebra
using SparseArrays
using DataStructures
using IncompleteLU
using Preconditioners
using Statistics
using IterativeSolvers
include("common.jl")

export diagonalapprox

# find apprimate inverse using ilu
function z_approx_ilu(A::AbstractMatrix, v)
    LU = IncompleteLU.ilu(A)
    L = LU.L + I
    U = LU.U
    X = U \ (L \ v)
    return X
end

function z_approx_chol(A::AbstractMatrix, v)
    P = CholeskyPreconditioner(A)
    return (P.L \ v) , P
end

# Minimize this function for value of t
# i < t < j, t ∈ 𝐙
function t_bisection(M, i, j)
    vals = []
    for t in i+1:j-1
        push!(vals, abs((M[i] - M[j])*(i - j) - (M[i] - M[t])*(i-t) - (M[t] - M[j])*(t-j)))
    end
    return argmin(vals)+i
end

function middle_index!(S, Q = Nothing, Mₒ = Nothing)
    temp = 0
    L = 0
    R = 0
    interval = 0
    for i in S
        if temp != 0
            if interval < i - temp
                interval = i - temp
                L = temp
                R = i
            end
            temp = i
        else
            temp = i
        end
    end
    t = Int64(ceil((L + R)/2))
    push!(S, t)
    if Q != Nothing
        delete!(Q, (L, R))
        if t - L > 1
            enqueue!(Q, (L, t), abs(sum(Mₒ[L:t]) - (Mₒ[L] + Mₒ[t])*((t-L))/2))
        end
        if R - t > 1
            enqueue!(Q, (t, R), abs(sum(Mₒ[t:R]) - (Mₒ[t] + Mₒ[R])*((R-t))/2))
        end
    end
end

function point_identification(M, maxPts)
    N = size(M, 1)
    Mₒ = sort(M)
    J = copy(M)
    numSamples = 1
    initErr = abs(sum(Mₒ) - (Mₒ[1] + Mₒ[N])*(N/2))
    tempErr = initErr
    Sₒ = SortedSet{Int64}()
    push!(Sₒ, 1)
    push!(Sₒ, N)
    Q = PriorityQueue{Tuple{Int64, Int64}, Float64}(Base.Order.Reverse)
    enqueue!(Q, (1, N), initErr)
    while numSamples < maxPts && peek(Q)[2] > 0.001*initErr
        # pop interval (L, R) with the largest error from Q
        interval = dequeue!(Q)
        L, R = interval[1], interval[2]

        # for t = L+1:R-1 find bisecting index t
        t = t_bisection(Mₒ, L, R)

        # Add t to Sₒ and increase numsamples
        if t - L >= 1 && R - t >= 1
            push!(Sₒ, t)
            numSamples = numSamples + 1
        end

        # Push interval (L, t) and (t, R) with temperror into Q
        if t - L > 1
            enqueue!(Q, (L, t), abs(sum(Mₒ[L:t]) - (Mₒ[L] + Mₒ[t])*((t-L)/2)))
        end
        if R - t > 1
            enqueue!(Q, (t, R), abs(sum(Mₒ[t:R]) - (Mₒ[t] + Mₒ[R])*((R-t)/2)))
        end
        # insert midpoint into largest interval
        if numSamples%5 == 0
            middle_index!(Sₒ, Q, Mₒ)
            numSamples = numSamples + 1
        end
    end

    # insert midpoints in the largest intervals
    while numSamples < maxPts
        middle_index!(Sₒ)
        numSamples = numSamples + 1
    end
    return Sₒ
end

function basis_vector!(e, k)
    e .= Ref(0)
    e[k] = one(eltype(e))
end

function linreg(X, Y)
    n = size(Y, 1)
    sX = sum(X)
    sY = sum(Y)
    sXY = sum(X.*Y)
    sX2 = sum(X.^2)

    c = (sY*sX2 - sX*sXY)/(n*sX2 - sX^2)
    b = (n*sXY - sX*sY)/(n*sX2 - sX^2)
    return b, c
end

function linear_model(S, D, M, n)
    b, c = linreg(S, D)
    ## Compute trace estimation Tf
    Tf = zero(eltype(D))
    for i in 1:n
        Tf += b * M[i] + c
    end
    return Tf
end

function diagonalapprox(A::AbstractMatrix, n::Int64, p::Int64)
    # Compute M = diag(approximation of A⁻¹)
    v = Matrix{eltype(A)}(undef, size(A,1), n)
    rademacherDistribution!(v)
    Z, pl = z_approx_chol(A, v)
    M = vec(mean(v .* Z , dims=2))

    # Compute fitting sample S, set of k indices
    S = collect(point_identification(M, p))

    # Solve for D(i) = e'(i) A⁻¹ e(i) for i = 1 ... k
    e = zeros(size(A, 1))
    D = eltype(A)[]
    for i in 1:size(S, 1)
        basis_vector!(e, S[i])
        push!(D, e' * cg(A, e, Pl = pl))
    end

    # Obtain fitting model f(M) ≈ D by fitting f(M(S)) to D(S)
    Tr = linear_model(S, D, M, size(A, 1))

    println(Tr)

end
