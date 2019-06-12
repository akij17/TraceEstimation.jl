# Stochastic Chebyshev Polynomial method
# Based on APPROXIMATING SPECTRAL SUMS OF LARGE-SCALE MATRICES USING STOCHASTIC CHEBYSHEV APPROXIMATIONS By Han, Malioutov, Avron and Shin

export ChebyHutchSpace, chebyhutch

using LinearAlgebra
using Parameters

𝓍(k, n) = cos((π * (k + 0.5))/(n+1))

struct ChebyHutchSpace{elt, TM, FN<:Function, FN2<:Function, TA<:AbstractArray{elt, 1}, TV<:AbstractVecOrMat{elt}, I<:Integer}
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

function ChebyHutchSpace(A, a, b; fn::Function=invfun, dfn::Function=rademacherDistribution!, m = 4, n = 6)
    elt = eltype(A)
    s = size(A, 1)
    C = elt[]
    v = Matrix{elt}(undef, s, m)
    w₀ = similar(v)
    w₁ = similar(v)
    w₂ = similar(v)
    u = similar(v)
    return ChebyHutchSpace(A, a, b, fn, dfn, C, w₀, w₁, w₂, v, u, m, n)
end

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

function chebyhutch(w::ChebyHutchSpace)
    @unpack A, a, b, C, fn, dfn, v, u, w₀, w₁, w₂, m, n = w
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
    # Allocation-free batch dot product and averaging
    return dot(v, u) / m
end

function chebyhutch(A; fn::Function=invfun, dfn::Function=rademacherDistribution!, m = 4, n = 6)
    # Estimate eigmax and eigmin for Chebyshev bounds
    mval = Int64(ceil(log(0.5/(1.648 * sqrt(size(A, 1))))/(-2 * sqrt(0.01))))
    w = SLQWorkspace(A, fn = fn, dfn = dfn, m = mval)
    dfn(w.v)
    w.v .= w.v ./ norm(w.v)
    lcz(w)
    λₘ = eigmax(w.T)
    λ₁ = eigmin(w.T)

    wx = ChebyHutchSpace(A, λₘ, λ₁, fn=fn, dfn=dfn, m = m, n = n)
    chebyhutch(wx)
end
