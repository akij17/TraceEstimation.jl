# Stochastic Lanczos Quadrature method
# Based on Ubaru, Shashanka, Jie Chen, and Yousef Saad.
# "Fast estimation of tr(F(A)) via stochastic lanczos quadrature, 2016."
# URL: http://www-users.cs.umn.edu/~saad/PDF/ys-2016-04.pdf.

using LinearAlgebra
using Parameters

# Predefined function and values
invfun(x) = 1/x
ϵ = 0.5
tol = 0.01

mutable struct SLQWorkspace{elt, TM<:AbstractMatrix{elt}, FN<:Function,
    FN2<:Function, I<:Integer, TV<:AbstractVector{elt},
    TS<:SymTridiagonal, TM2<:AbstractMatrix{elt}, R<:Real}
    A::TM
    fn::FN
    rfn::FN2
    m::I
    v::TV
    nᵥ::I
    ctol::R
    T::TS
    α::TV
    β::TV
    ω::TV
    Y::TM2
    Θ::TV
    v₀::TV
    result::elt
end

"""
    SLQWorkspace(A::AbstractMatrix; fn::Function, rfn::Function, ctol, m, nv)

Create an SLQWorkspace for supplied SPD Matrix A.
Use it to calculate tr(fn(A)).

# Arguments
- `A` : Symmetric Positive Definite Matrix
- `fn` : Function to apply. By default uses inverse function
- `rfn` : Random value generator for Rademacher Distribution. By default uses
            Base.rand
- `ctol` : SLQ Convergence Tolerance value. By default ctol = 0.1
- `m` : Specify value for lanczos steps. By default m = 15
- `nv` : Specify value for SLQ iterations. By default nb = 10
"""
function SLQWorkspace(A; fn::Function=invfun, rfn::Function=Base.rand,
     ctol=0.1, m=15, nv=10)
    elt = eltype(A)
    Atype = typeof(A)
    n = size(A, 1)
    v = similar(A, n)
    v₀ = similar(v)
    α = similar(A, m)
    β = similar(A, m-1)
    ω = similar(A, n)
    Y = similar(A, m, m)
    Θ = similar(A, m)
    T = SymTridiagonal(α, β)
    result = zero(elt)
    return SLQWorkspace(A, fn, rfn, m, v, nv, ctol, T, α, β, ω, Y, Θ, v₀, result)
end

function rademacherDistribution!(v, rfn::Function, t::Type)
    o = one(t)
    v .= rfn.(Ref(-o:2*o:o))
end

function lcz(w::SLQWorkspace)
    α₀ = zero(eltype(w.A))
    β₀ = zero(eltype(w.A))
    fill!(w.v₀, 0)
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
end
"""
    slq(w::SLQWorkspace; skipverify = false)
    slq(A::AbstractMatrix; skipverify = false, fn::Function = invfun,
    rfn::Function = Base.rand, ctol = 0.1, eps = ϵ, mtol = tol)

SLQ method to calculate tr(fn(A)) for a Symmetric Positive Definite matrix
A and an analytic function fn.

# Arguments
- `A` : Symmetric Positive Definite Matrix
- `skipverify` : Skip isposdef(A) verification. By default skipverify = false
- `fn` : Function to apply on A before trace calculation. fn must be analytic
            λₘ and λ₁ of A. By default fn = inv
- `rfn` : Random number generator. Should be able to take ranges. By default
            rfn = Base.rand, example, Base.rand.(Ref(-1:2:1, n))
- `ctol` : SLQ Convergence Tolerance value. By default ctol = 0.1
- `eps` : Error bound for lanczos steps calculation. By default eps = 0.5
- `mtol` : Tolerance for eigenvalue Convergence. By default mtol = 0.01
"""
function slq(w::SLQWorkspace; skipverify = false)
    if skipverify || isposdef(w.A)
        tr = zero(eltype(w.A))
        for i in 1:w.nᵥ
            @unpack A, v, T, Y, Θ, m, nᵥ, result, ctol, fn, rfn = w
            rademacherDistribution!(v, rfn, eltype(A))
            v .= v ./ norm(v)
            lcz(w)
            Y .= eigvecs(T)
            Θ .= eigvals(T)
            for j in 1:m
                τ = Y[1,j]
                tr = tr + τ^2 * fn(Θ[j])
            end
            if isapprox(result, tr, rtol = ctol)
                @show w.nᵥ = i
                break
            end
            result = tr
            @pack! w = A, v, T, Y, Θ, m, nᵥ, result, ctol, fn, rfn
        end
        tr = size(w.A, 1)/w.nᵥ * tr
    else
        println("Given Matrix is NOT Symmetric Positive Definite")
    end
end

function slq(A::AbstractMatrix; skipverify = false, fn::Function = invfun,
    rfn::Function = Base.rand, ctol = 0.1, eps = ϵ, mtol = tol)

    # Estimate eigmax and eigmin for SLQ bounds
    mval = Int64(ceil(log(eps/(1.648 * sqrt(size(A, 1))))/(-2 * sqrt(mtol))))
    w = SLQWorkspace(A, fn = fn, rfn = rfn, m = mval)
    rademacherDistribution!(w.v, w.rfn, eltype(w.A))
    w.v .= w.v ./ norm(w.v)
    lcz(w)
    λₘ = eigmax(w.T)
    λ₁ = eigmin(w.T)

    if λ₁ < 1 && λₘ > 1
        @warn "Eigenvalues cross zero. Functions like log may not give
        correct results. Try scaling the input matrix."
    end

    # SLQ bounds
    κ = λₘ/λ₁
    Mₚ = fn(λₘ)
    mₚ = fn(λ₁)
    ρ = (sqrt(κ) + 1)/(sqrt(κ) - 1)
    K = ((λₘ - λ₁) * (sqrt(κ) - 1)^2 * Mₚ)/(sqrt(κ) * mₚ)
    @show mval = Int64(ceil((sqrt(κ)/4) * log(K/eps)))
    @show nval = Int64(ceil((24/ϵ^2) * log(2/mtol)))

    # Re-construct SLQWorkspace
    w = SLQWorkspace(A, fn = fn, rfn = rfn, m = mval, nv = nval, ctol = ctol)
    slq(w, skipverify = skipverify)
end
#=
global X = (rand(5095,5095))
X = X + X'
while !isposdef(X)
    global X
    X = X + 10I
end

sqrtfun(x) = sqrt(x)
logfun(x) = log(x)
#w = SLQWorkspace(X, fn = logfun)
#@time nothing; @time slq(w, skipverify = true)

println(slq(X))
println(tr(inv(X)))
=#
