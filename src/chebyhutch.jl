# Stochastic Chebyshev Polynomial method
# Based on APPROXIMATING SPECTRAL SUMS OF LARGE-SCALE MATRICES USING STOCHASTIC CHEBYSHEV APPROXIMATIONS By Han, Malioutov, Avron and Shin

export ChebyHutchSpace, chebyhutch, chebydiagonal

using LinearAlgebra
using Parameters
using Statistics

# Lanczos iteration for finding extremal eigenvalues of the matrix
function lczeigen(A, fn, dfn)
    # Estimate eigmax and eigmin for Chebyshev bounds
    mval = Int64(ceil(log(ϵ/(1.648 * sqrt(size(A, 1))))/(-2 * sqrt(ξ))))
    w = SLQWorkspace(A, fn = fn, dfn = dfn, m = mval)
    dfn(w.v)
    w.v .= w.v ./ norm(w.v)
    lcz(w)
    λₘ = eigmax(w.T)
    λ₁ = eigmin(w.T)
    return λ₁, λₘ
end

# Chebyshev polynomial method helper functions
𝓍(k, n) = cos((π * (k + 0.5))/(n+1))

function T(j, x, Tvals)
    if j == 0
        return 1
    end
    if j == 1
        return x
    end
    if haskey(Tvals, j)
        return Tvals[j]
    else
        Tvals[j] = (2 * x * T(j-1, x, Tvals)) - T(j-2, x, Tvals)
        return Tvals[j]
    end
end

function coeff(j, n, a, b, fn)
    fs = zero(eltype(a))
    for k in 0:n
        x = 𝓍(k, n)
        Tvals = Dict{Int, Float64}()
        fs = fs + fn((((b-a)/2) * x) + (b+a)/2) * T(j, x, Tvals)
    end
    if j == 0
        return (1/(n+1)) * fs
    end
    return (2/(n+1)) * fs
end

mutable struct ChebyHutchSpace{elt, TM, FN<:Function, FN2<:Function, TA<:AbstractArray{elt, 1}, TV<:AbstractVecOrMat{elt}, I<:Int64}
    A::TM
    a::elt
    b::elt
    fn::FN
    dfn::FN2
    C::TA
    w₀::TV
    w₁::TV
    w₂::TV
    v::TV
    u::TV
    m::I
    n::I
end
"""
    ChebyHutchSpace(A::AbstractMatrix, a::Number, b::Number; fn::Function=invfun, dfn::Function=rademacherDistribution!, m = 4, n = 6)

Create Chebyshev-Hutchinson Workspace for chebyhutch or chebydiagonal methods.

# Arguments
- `A` : Symmetric Positive Definite Matrix
- `a` : Bound for minimum eigenvalue of A
- `b` : Bound for maximum eigenvalue of A
- `fn` : Function to appy. By default uses inverse function.
- `dfn` : Distribution function that returns a vector v. By default uses rademacherDistribution!
- `m` : Iteration number, increase this for precision. By default m = 4
- `n` : Polynomial degree, increase this for accuracy. By default n = 6
"""
function ChebyHutchSpace(A::AbstractMatrix, a::Number, b::Number; fn::Function=invfun, dfn::Function=rademacherDistribution!, m = 4, n = 6)
    elt = eltype(A)0
    s = size(A, 1)
    C = elt[]
    v = Matrix{elt}(undef, s, min(m, blocksize))
    w₀ = similar(v)
    w₁ = similar(v)
    w₂ = similar(v)
    u = similar(v)
    return ChebyHutchSpace(A, a, b, fn, dfn, C, w₀, w₁, w₂, v, u, m, n)
end

function chebypm(w::ChebyHutchSpace)
    @unpack A, a, b, C, fn, dfn, v, u, w₀, w₁, w₂, m, n = w
    tr = zero(eltype(A))
    for j in 0:n
        push!(C, coeff(j, n, a, b, fn))
    end
    dfn(v)
    w₀ .= v
    mul!(w₁, A, v)
    rmul!(w₁, 2/(b-a))
    w₁ .= w₁ .- (((b+a)/(b-a)) .* v)
    u .= (C[1] .* w₀) .+ (C[2] .* w₁)
    for j in 2:n
        mul!(w₂, A, w₁)
        rmul!(w₂, 4/(b-a))
        w₂ .= w₂ .- ((2(b+a)/(b-a)) .* w₁) .- w₀
        u .= u .+ (C[j+1] .* w₂)
        w₀ .= w₁
        w₁ .= w₂
    end
    return v, u, m
end
"""
    chebyhutch(w::ChebyHutchSpace)
    chebyhutch(A::AbstractMatrix, m::Integer, n::Integer; fn::Function=invfun, dfn::Function=rademacherDistribution!)
    chebyhutch(A::AbstractMatrix; fn::Function=invfun, dfn::Function=rademacherDistribution!)

Chebyshev-Hutchinson to estimate tr(fn(A)), for given matrix A and an analytic function fn.

# Arguments
- `A` : Symmetric Positive Definite Matrix
- `m` : Iteration number, increase this for precision. By default m = 4
- `n` : Polynomial degree, increase this for accuracy. By default n = 6
- `fn` : Function to appy. By default uses inverse function.
- `dfn` : Distribution function that returns a vector v. By default uses rademacherDistribution!
"""
function chebyhutch(w::ChebyHutchSpace)
    v, u, m = chebypm(w)
    return dot(v, u) / m
end

function chebyhutch(A::AbstractMatrix, m::Integer, n::Integer; fn::Function=invfun, dfn::Function=rademacherDistribution!)
    # calculate extremal eigenvals
    λ₁, λₘ = lczeigen(A, fn, dfn)

    wx = ChebyHutchSpace(A, λₘ, λ₁, fn=fn, dfn=dfn, m = m, n = n)
    return chebyhutch(wx)
end

function chebyhutch(A::AbstractMatrix; fn::Function=invfun, dfn::Function=rademacherDistribution!)
    # calculate extremal eigenvals
    λ₁, λₘ = lczeigen(A, fn, dfn)

    # calculate values of m and n
    # these bounds are for theoretical purposes only
    κ = λₘ/λ₁
    ρ = sqrt(2 * κ - 1) - 1
    mVal = Int64(ceil(54 * (ϵ)^(-2) * log(2/ξ)/16))
    nVal = Int64(ceil((log(8/ϵ)*ρ*κ)/(log((2/ρ) + 1))/16))

    wx = ChebyHutchSpace(A, λₘ, λ₁, fn=fn, dfn=dfn, m = mVal, n = nVal)
    return chebyhutch(wx)
end
"""
    chebydiagonal(w::ChebyHutchSpace)
    chebydiagonal(A::AbstractMatrix, m::Integer, n::Integer; fn::Function=invfun, dfn::Function=rademacherDistribution!)

Chebyshev-Hutchinson to estimate diagonal elements of fn(A), for given matrix A and an analytic function fn.

# Arguments
- `A` : Symmetric Positive Definite Matrix
- `m` : Iteration number, increase this for precision. By default m = 4
- `n` : Polynomial degree, increase this for accuracy. By default n = 6
- `fn` : Function to appy. By default uses inverse function.
- `dfn` : Distribution function that returns a vector v. By default uses rademacherDistribution!
"""
function chebydiagonal(w::ChebyHutchSpace)
    v, u, m = chebypm(w)
    return vec(mean(v .* u, dims=2))
end

function chebydiagonal(A, m, n; fn::Function=invfun, dfn::Function=rademacherDistribution!)
    # calculate extremal eigenvals
    @time @show λ₁, λₘ = lczeigen(A, fn, dfn)
    wx = ChebyHutchSpace(A, λₘ, λ₁, fn=fn, dfn=dfn, m = m, n = n)
    return chebydiagonal(wx)
end
