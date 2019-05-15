# Stochastic Lanczos Quadrature method

using LinearAlgebra
using Parameters

# Predefined function
invfun(x) = 1/x

mutable struct SLQWorkspace{elt, TM<:AbstractMatrix{elt}, FN<:Function,
    I<:Integer, TV<:AbstractVector{elt}, TS<:SymTridiagonal, R<:Real}
    A::TM
    f::FN
    m::I
    v::TV
    nᵥ::I
    ctol::R
    T::TS
    α::TV
    β::TV
    ω::TV
    Y::TM
    Θ::TV
    v₀::TV
    result::R
end

function rademacherDistribution!(v, t::Type)
    o = one(t)
    v .= Base.rand.(Ref(-o:2*o:o))
end

function SLQWorkspace(A; f::Function=invfun, ctol=0.1, m=15, nv=10)
    elt = eltype(A)
    n = size(A, 1)
    v = zeros(elt, n)
    v₀ = zeros(elt, n)
    α = zeros(elt, m)
    β = zeros(elt, m-1)
    ω = zeros(elt, n)
    Y = zeros(elt, m, m)
    Θ = zeros(elt, m)
    T = SymTridiagonal(α, β)
    result = zero(eltype(A))
    return SLQWorkspace(A, f, m, v, nv, ctol, T, α, β, ω, Y, Θ, v₀, result)
end

function lcz(w::SLQWorkspace)
    α₀ = zero(eltype(w.A))
    β₀ = zero(eltype(w.A))
    for i in 1:w.m
        @unpack A, v, ω, v₀, α, β, m, T = w
        mul!(ω, A, v)
        α₀ = dot(ω, v)
        ω .= ω .- (α₀ .* v) .- (β₀ .* v₀)
        β₀ = norm(ω)

        α[i] = α₀
        if i < m
            β[i] = β₀
        end

        copy!(v₀, v)
        v .= ω ./ β₀
        @pack! w = A, v, ω, v₀, α, β, m, T
    end
    w.T = SymTridiagonal(w.α, w.β)
    return w.T
end

function slq(w::SLQWorkspace; skipverify = false)
    if skipverify || isposdef(w.A)
        tr = zero(eltype(w.A))
        for i in 1:w.nᵥ
            @unpack A, v, T, Y, Θ, m, nᵥ, result, ctol, f = w
            rademacherDistribution!(v, eltype(A))
            v .= v ./ norm(v)
            lcz(w)
            Y .= eigvecs(T)
            Θ .= eigvals(T)
            for j in 1:m
                τ = Y[1,j]
                tr = tr + τ^2 * f(Θ[j])
            end
            if isapprox(result, tr, rtol = ctol)
                w.nᵥ = i
                break
            end
            result = tr
            @pack! w = A, v, T, Y, Θ, m, nᵥ, result, ctol, f
        end
        tr = size(w.A, 1)/w.nᵥ * tr
    else
        println("Given Matrix is NOT Symmetric Positive Definite")
    end
end

global X = rand(500,500)
X = X + X'
while !isposdef(X)
    global X
    X = X + 10I
end

sqrtfun(x) = sqrt(x)
w = SLQWorkspace(X, f = sqrtfun)
@time slq(w, skipverify = true)
