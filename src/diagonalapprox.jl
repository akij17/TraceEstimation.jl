# Diagonal Approximation algorithm

using LinearAlgebra
using SparseArrays
using DataStructures
using Preconditioners
using Statistics
using IterativeSolvers
include("common.jl")

export diagonalapprox

struct _pchip
    N :: Integer
    xs :: Array{Float64}
    ys :: Array{Float64}
    ds :: Array{Float64}
end

ϕ(t) = 3t^2 - 2t^3
ψ(t) = t^3 - t^2

function _interp(pchip :: _pchip, x :: Number)
    i = _pchip_index(pchip, x)
    x1, x2 = pchip.xs[i:i+1]
    y1, y2 = pchip.ys[i:i+1]
    d1, d2 = pchip.ds[i:i+1]
    h = x2 - x1

    (y1 * ϕ((x2-x)/h)
     + y2 * ϕ((x-x1)/h)
     - d1*h * ψ((x2-x)/h)
     + d2*h * ψ((x-x1)/h))
end

function _pchip_index(pchip :: _pchip, x)
    N = pchip.N
    if N < 200  # Approximate performance cross-over on my old intel i7-3517U
        i = _pchip_index_linear_search(pchip, x)
    else
        i = _pchip_index_bisectional_search(pchip, x)
    end
    if i == N
        # Treat right endpoint as part of rightmost interval
        @assert (x ≈ pchip.xs[N])
        i = N-1
    end
    i
end

function _pchip_index_linear_search(pchip :: _pchip, x)
    xmin = pchip.xs[1]
    @assert (x >= xmin)

    i = 1
    N = pchip.N
    while i < N  &&  x >= pchip.xs[i+1]
        i = i + 1
    end
    i
end

function _pchip_index_bisectional_search(pchip :: _pchip, x)
    N = pchip.N
    imin, imax = 1, N
    xmin = pchip.xs[imin]
    xmax = pchip.xs[imax]
    @assert (x >= xmin && x <= xmax)

    i = imin + div(imax - imin + 1, 2)
    while imin < imax
        if x < pchip.xs[i]
            imax = i - 1
        elseif x >= pchip.xs[i+1]
            imin = i + 1
        else
            break
        end
        i = imin + div(imax - imin + 1, 2)
    end
    i
end

function _initial_ds_scipy(xs, ys)
    h(i) = xs[i+1]-xs[i]
    Δ(i) = (ys[i+1]-ys[i]) / h(i)

    N = length(xs)
    ds = Vector{Float64}(undef, size(xs, 1))
    if N == 2
        ds[:] .= Δ(1)
    else
        Δl = Δ(1)
        hl = h(1)
        for i ∈ 2:N-1
            Δr = Δ(i)
            hr = h(i)
            if sign(Δl) != sign(Δr) || Δl ≈ 0.0 || Δr ≈ 0.0
                ds[i] = 0.0
            else
                wl = 2hl + hr
                wr = hl + 2hr
                axx = (wl + wr)
                bxx = (wl/Δl + wr/Δr)
                ds[i] =  axx / bxx
            end
            Δl = Δr
            hl = hr
        end
        ds[1] = _edge_derivative(h(1), h(2), Δ(1), Δ(2))
        ds[N] = _edge_derivative(h(N-1), h(N-2), Δ(N-1), Δ(N-2))
    end
    ds
end

function _edge_derivative(h1, h2, Δ1, Δ2)
    d = ((2h1 + h2)*Δ1 - h2*Δ2) / (h1 + h2)
    if sign(d) != sign(Δ1)
        d = 0.0
    elseif sign(Δ1) != sign(Δ2)  &&  abs(d) > abs(3Δ1)
        d = 3Δ1
    end
    d
end

function interpolate(xs, ys)
    xs_ = [x for x ∈ xs]
    ys_ = [y for y ∈ ys]
    _assert_xs_ys(xs_, ys_)
    ds = _initial_ds_scipy(xs_, ys_)
    pchip = _pchip(length(xs_), xs_, ys_, ds)

    x -> _interp(pchip, x)
end

function _assert_xs_ys(xs, ys)
    N = length(xs)
    @assert (N > 1)
    @assert (N == length(ys))
    assert_monotonic_increase(xs)
end

function assert_monotonic_increase(xs)
    foldl((a,b) -> (@assert (a < b); b), xs)
end


# find approximate inverse using cholesky
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

# Create new index in the middle of the longest L-R range
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
    J = sortperm(M)
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

        # for t = L+1:R-1 find bisection index t
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

        # insert midpoint into largest interval after every 5 samples
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

    ## unique Mₒ values and Sₒ indices
    temp = Set()
    S = []
    for i in Sₒ
        if (Mₒ[i] in temp) == false
            push!(S, i)
            push!(temp, Mₒ[i])
        end
    end
    S2 = []
    for i in S
        push!(S2, J[i])
    end
    return S2
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
    b, c = linreg(M[S], D)
    ## Compute trace estimation Tf
    Tf = zero(eltype(D))
    for i in 1:n
        Tf += b * M[i] + c
    end
    return Tf
end

function pchip_iterpolation(S, M, D)
    Ms = Vector{eltype(D)}(undef, size(S, 1))
    Ms .= M[S]
    pchip = interpolate(Ms, D)
    Tf = zero(eltype(D))
    for i in 1:size(M, 1)
        Tf += pchip(M[i])
    end
    return Tf
end

function diagonalapprox(A::AbstractMatrix, n::Int64, p::Int64; fitmodel = "linear")
    # Compute M = diag(approximation of A⁻¹)
    v = Matrix{eltype(A)}(undef, size(A,1), n)
    rademacherDistribution!(v)
    Z, pl = z_approx_chol(A, v)
    M = vec(mean(v .* Z , dims=2))

    # Compute fitting sample S, set of k indices
    #S = collect(point_identification(M, p))
    S = point_identification(M, p)

    # Solve for D(i) = e'(i) A⁻¹ e(i) for i = 1 ... k
    e = zeros(size(A, 1))
    D = eltype(A)[]
    for i in 1:size(S, 1)
        basis_vector!(e, S[i])
        push!(D, e' * cg(A, e, Pl = pl))
    end

    # Obtain fitting model f(M) ≈ D by fitting f(M(S)) to D(S)
    if fitmodel == "pchip"
        println(pchip_iterpolation(S, M, D))
    else
        println(linear_model(S, D, M, size(A, 1)))
    end
end
