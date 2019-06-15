# Stochastic Chebyshev Polynomial method
# Based on APPROXIMATING SPECTRAL SUMS OF LARGE-SCALE MATRICES USING STOCHASTIC CHEBYSHEV APPROXIMATIONS By Han, Malioutov, Avron and Shin

export ChebyHutchSpace, chebyhutch

using LinearAlgebra
using Parameters
include("common.jl")
include("slq.jl")

const ϵ = 0.5
const ξ = 0.1

function lczeigen(A, fn, dfn)
    # Estimate eigmax and eigmin for Chebyshev bounds
    mval = Int64(ceil(log(0.5/(1.648 * sqrt(size(A, 1))))/(-2 * sqrt(0.01))))
    w = SLQWorkspace(A, fn = fn, dfn = dfn, m = mval)
    dfn(w.v)
    w.v .= w.v ./ norm(w.v)
    lcz(w)
    λₘ = eigmax(w.T)
    λ₁ = eigmin(w.T)
    return λ₁, λₘ
end

𝓍(k, n) = cos((π * (k + 0.5))/(n+1))

struct ChebyHutchSpace{elt, TM, FN<:Function, FN2<:Function, TA<:AbstractArray{elt, 1}, TV<:AbstractVecOrMat{elt}, I<:Int64}
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
    blocksize::I
end

function ChebyHutchSpace(A, a, b; fn::Function=invfun, dfn::Function=rademacherDistribution!, m = 4, n = 6, blocksize = m)
    elt = eltype(A)
    s = size(A, 1)
    C = elt[]
    v = Matrix{elt}(undef, s, min(m, blocksize))
    w₀ = similar(v)
    w₁ = similar(v)
    w₂ = similar(v)
    u = similar(v)
    return ChebyHutchSpace(A, a, b, fn, dfn, C, w₀, w₁, w₂, v, u, m, n, blocksize)
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
    @unpack A, a, b, C, fn, dfn, v, u, w₀, w₁, w₂, m, n, blocksize = w
    tr = zero(eltype(A))
    for j in 0:n
        push!(C, coeff(j, n, a, b, fn))
    end
    for i in 0:blocksize:m-1
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
        tr = tr + dot(v, u) / m
    end
    return tr
end

function chebyhutch(A, m, n; fn::Function=invfun, dfn::Function=rademacherDistribution!, blocksize = m)
    # calculate extremal eigenvals
    λ₁, λₘ = lczeigen(A, fn, dfn)

    wx = ChebyHutchSpace(A, λₘ, λ₁, fn=fn, dfn=dfn, m = m, n = n, blocksize = m)
    chebyhutch(wx)
end

function chebyhutch(A; fn::Function=invfun, dfn::Function=rademacherDistribution!)
    # calculate extremal eigenvals
    λ₁, λₘ = lczeigen(A, fn, dfn)

    # calculate values of m and n
    # these bounds are for theoretical purposes only
    κ = λₘ/λ₁
    ρ = sqrt(2 * κ - 1) - 1
    mVal = Int64(ceil(54 * (ϵ)^(-2) * log(2/ξ)/16))
    nVal = Int64(ceil((log(8/ϵ)*ρ*κ)/(log((2/ρ) + 1))/16))

    wx = ChebyHutchSpace(A, λₘ, λ₁, fn=fn, dfn=dfn, m = mVal, n = nVal, blocksize = mVal)
    chebyhutch(wx)
end
#=
using CuArrays, TopOpt
s = (40, 10) # increase to increase the matrix size
xmin = 0.9 # decrease to increase the condition number
problem = HalfMBB(Val{:Linear}, s, (1.0, 1.0), 1.0, 0.3, 1.0)
solver = FEASolver(Displacement, Direct, problem, xmin = xmin)
n = length(solver.vars)
solver.vars[rand(1:n, n÷2)] .= 0
solver()
K = solver.globalinfo.K

A = KMatrix(K)
=#
struct KMatrix{T, M<:AbstractMatrix{T},Md<:AbstractMatrix{T}, V<:AbstractVector{T}} <:AbstractMatrix{T}
    K::M
    invD::Md
    temp::V
end
function KMatrix(K::AbstractMatrix)
    invD = inv(Diagonal(sqrt.(diag(K))))
    temp = Vector{eltype(K)}(undef, size(K,1))
    return KMatrix(K, invD, temp)
end
Base.size(A::KMatrix, i...) = size(A.K, i...)
Base.getindex(A::KMatrix, i...) = getindex(A.K, i...)
LinearAlgebra.mul!(y::AbstractVector{T}, A::KMatrix, x::AbstractVector{T}) where {T} = (mul!(y, A.invD, x); mul!(A.temp, A.K, y); mul!(y, A.invD, A.temp))

function chebybreak(A, m, n; fn::Function=invfun, dfn::Function=rademacherDistribution!, blocksize = m)
    # calculate extremal eigenvals
    λ₁, λₘ = lczeigen(A, fn, dfn)

    wx = ChebyHutchSpace(A, λₘ, λ₁, fn=fn, dfn=dfn, m = m, n = n, blocksize = m)
    return chebybreak(wx)
end

function chebybreak(w::ChebyHutchSpace)
    @unpack A, a, b, C, fn, dfn, v, u, w₀, w₁, w₂, m, n, blocksize = w
    tr = zero(eltype(A))
    for j in 0:n
        push!(C, coeff(j, n, a, b, fn))
    end
    for i in 0:blocksize:m-1
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
        #tr = tr + dot(v, u) / m
    end
    u[:,m] .= v[:,m] .* u[:,m]
    return Diagonal(u[:,m])
end

A = rand(10,10)
A = A + A'
A = A + 20I
println(diag(chebybreak(A,8,12)))
println(diag(inv(A)))
