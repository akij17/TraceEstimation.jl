# Stochastic Lanczos Quadrature method
# Based on Ubaru, Shashanka, Jie Chen, and Yousef Saad.
# "Fast estimation of tr(F(A)) via stochastic lanczos quadrature, 2016."
# URL: http://www-users.cs.umn.edu/~saad/PDF/ys-2016-04.pdf.

export SLQWorkspace, slq

using LinearAlgebra
using Parameters

struct SLQWorkspace{elt, TM<:AbstractMatrix{elt}, FN<:Function,
    FN2<:Function, I<:Integer, TV<:AbstractVector{elt}, AV<:AbstractVector{elt},
    TS<:SymTridiagonal, TM2<:AbstractMatrix{elt}, R<:Real}
    A::TM
    fn::FN
    dfn::FN2
    m::I
    v::AV
    nᵥ::I
    ctol::R
    T::TS
    α::TV
    β::TV
    ω::TV
    Y::TM2
    Θ::TV
    v₀::AV
end

# Using v = Vector{elt}(undef, n) will increase the performance but due to
# support issue for CuArray, it must be kept similar(A, n).
# This is a good use-case for full (which was removed in Julia 1.0) where
# we could just get a dense vector from a sparse one solving the performance
# issue
"""
    SLQWorkspace(A::AbstractMatrix; fn::Function, dfn::Function, ctol, m, nv)

Create an SLQWorkspace for supplied SPD Matrix A.
Use it to calculate tr(fn(A)).

# Arguments
- `A` : Symmetric Positive Definite Matrix
- `fn` : Function to apply. By default uses inverse function
- `dfn` : Distribution function for v (random dist. with norm(v) = 1). By default uses rademacherDistribution!(v::Vector, t::Type)
- `ctol` : SLQ Convergence Tolerance value. By default ctol = 0.1
- `m` : Specify value for lanczos steps. By default m = 15
- `nv` : Specify value for SLQ iterations. By default nb = 10
"""
function SLQWorkspace(A; fn::Function=invfun, dfn::Function=rademacherDistribution!, ctol=0.1, m=15, nv=10)
    elt = eltype(A)
    Atype = typeof(A)
    n = size(A, 1)
    #v = Vector{elt}(undef, n)
    v = similar(A, n)
    v₀ = similar(v)
    α = Vector{elt}(undef, m)
    β = Vector{elt}(undef, m-1)
    ω = Vector{elt}(undef, n)
    Y = similar(A, m, m)
    Θ = similar(α)
    T = SymTridiagonal(α, β)
    return SLQWorkspace(A, fn, dfn, m, v, nv, ctol, T, α, β, ω, Y, Θ, v₀)
end

function lcz(w::SLQWorkspace)
    α₀ = zero(eltype(w.A))
    β₀ = zero(eltype(w.A))
    fill!(w.v₀, 0)
    # Following loop executes lanczos steps
    @unpack A, v, ω, v₀, α, β, m, T = w
    for i in 1:w.m
        mul!(ω, A, v)
        α₀ = dot(ω, v)
        ω .= ω .- (α₀ .* v) .- (β₀ .* v₀)
        β₀ = norm(ω)

        α[i] = α₀
        if i < m
            β[i] = β₀
        end

        # Sparse Vectors do not support copy!
        #copy!(v₀, v)
        v₀ .= v
        v .= ω ./ β₀
    end
end

"""
    slq(w::SLQWorkspace; skipverify = false)
    slq(A::AbstractMatrix; skipverify = false, fn::Function = invfun,
    dfn::Function = Base.rand, ctol = 0.1, eps = ϵ, mtol = tol)

SLQ method to calculate tr(fn(A)) for a Symmetric Positive Definite matrix
A and an analytic function fn.

# Arguments
- `A` : Symmetric Positive Definite Matrix
- `skipverify` : Skip isposdef(A) verification. By default skipverify = false
- `fn` : Function to apply on A before trace calculation. fn must be analytic λₘ and λ₁ of A. By default fn = inv
- `dfn` : Distribution function for v (random dist. with norm(v) = 1). By default uses rademacherDistribution!(v::Vector, t::Type)
- `ctol` : SLQ Convergence Tolerance value. Decrease this for higher precision. By default ctol = 0.1
- `eps` : Error bound for lanczos steps calculation. Decrease this for higher accuracy. By default eps = 0.05
- `mtol` : Tolerance for eigenvalue Convergence. Decrease this for precision. By default mtol = 0.01
"""
function slq(w::SLQWorkspace; skipverify = false)
    @unpack A, v, T, Y, Θ, m, nᵥ, ctol, fn, dfn = w
    tr = zero(eltype(w.A))
    if skipverify || isposdef(w.A)
        actual_nᵥ = nᵥ
        for i in 1:w.nᵥ
            prev_tr = tr
            # Create a uniform random distribution with norm(v) = 1
            dfn(v)
            v .= v ./ norm(v)
            # Run lanczos algorithm to find estimate Ritz SymTridiagonal
            lcz(w)
            Y .= eigvecs(T)
            Θ .= eigvals(T)
            for j in 1:m
                τ = Y[1,j]
                tr = tr + τ^2 * fn(Θ[j])
            end
            if isapprox(prev_tr, tr, rtol = ctol)
                actual_nᵥ = i
                break
            end
        end
        tr = size(w.A, 1)/actual_nᵥ * tr
    else
        throw("Given Matrix is NOT Symmetric Positive Definite")
    end

    return tr
end

function slq(A::AbstractMatrix; skipverify = false, fn::Function = invfun, dfn::Function = rademacherDistribution!, ctol = 0.1, eps = ϵ, mtol = ξ)

    # Estimate eigmax and eigmin for SLQ bounds
    mval = Int64(ceil(log(eps/(1.648 * sqrt(size(A, 1))))/(-2 * sqrt(mtol))))
    w = SLQWorkspace(A, fn = fn, dfn = dfn, m = mval)
    #rademacherDistribution!(w.v)
    w.dfn(w.v)
    w.v .= w.v ./ norm(w.v)
    lcz(w)
    λₘ = eigmax(w.T)
    λ₁ = eigmin(w.T)

    if λ₁ < 1 && λₘ > 1
        @warn "Eigenvalues cross zero. Functions like log may not give correct results. Try scaling the input matrix."
    end

    # SLQ bounds
    # Todo: Research and create better bounds for λ₁ < 1 && λₘ > 1 case
    κ = λₘ/λ₁
    Mₚ = fn(λₘ)
    mₚ = fn(λ₁)
    ρ = (sqrt(κ) + 1)/(sqrt(κ) - 1)
    K = ((λₘ - λ₁) * (sqrt(κ) - 1)^2 * Mₚ)/(sqrt(κ) * mₚ)
    mval = Int64(ceil((sqrt(κ)/4) * log(K/eps)))
    if mval < 10
        @warn "Low lanczos step value. Try decreasing eps and mtol for better accuracy."
        mval = 5
    end
    nval = Int64(ceil((24/ϵ^2) * log(2/mtol)))

    # Re-construct SLQWorkspace
    w = SLQWorkspace(A, fn = fn, dfn = dfn, m = mval, nv = nval, ctol = ctol)
    slq(w, skipverify = skipverify)
end
