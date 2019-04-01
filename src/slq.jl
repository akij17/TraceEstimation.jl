# Stochastic Lanczos Trace Estimator

using LinearAlgebra
include("lanczos.jl")

export SLQWorkspace, slq

# predefined function
invfun(x) = 1/x

mutable struct SLQWorkspace{T, TM<:AbstractMatrix{T}, fn<:Function, rn <: Function,
    R<:Number, I<:Integer}
    A::TM
    f::fn
    randfunc::rn
    m::I
    nᵥ::I
    ctol::R
    result::T
end

"""
    SLQWorkspace(A; f::Function==invfun, ctol = 0.1, m = 50, nv = 30)
    SLQWorkspace(A, randfunc::Function; f::Function==invfun, ctol = 0.1, m = 50, nv = 30)

Create SLQWorkspace to work with Stochastic Lanczos Quadrature Algorithm.
# Arguments
 - `A` : Symmetric Positive Definite Matrix with low condition number
 - `f` : Function to apply while calculating trace
         (If none supplied, f(x) = invfun(X) = 1/x)
 - `randfunc` : Function to generate arbitary random vectors for Lanczos iteration
                (If none supplied, f(n) = Base.rand(-1:2:1, n))
 - `ctol` : Convergence Tolerance
 - `m` : Lanczos iterations
 - `nv` : SLQ Iterations (if convergence is achieved SLQ stops before nv)
"""
function SLQWorkspace(A; f::Function=invfun, ctol=0.1, m = 50, nv = 30)
    o = one(eltype(A) <: Integer ? Float64 : eltype(A))
    randfunc(n) = Base.rand(-o:2*o:o, n)
    return SLQWorkspace(A, f, randfunc, m, nv, ctol, zero(eltype(A)))
end

function SLQWorkspace(A, randfunc::Function; f::Function=invfun, ctol=0.1, m = 50, nv = 30)
    return SLQWorkspace(A, f, randfunc, m, nv, ctol, zero(eltype(A)))
end

"""
    slq(w::SLQWorkspace; skipverify = false)

Takes a SLQWorkspace object `w` and returns tr(f(w.A)).
# Arguments
 - w : SLQWorkspace (check example)
 - skipverify : Pass true only if you are sure matrix is SPD

Example:
w = SLQWorkspace(A, m = 20, nv = 10)
trace = slq(w)
"""
function slq(w::SLQWorkspace; skipverify = false)
    if skipverify || isposdef(w.A)
        tr = zero(eltype(w.A))
        for i in 1:w.nᵥ
            v₀ = w.randfunc(size(w.A, 1))
            v₀ = v₀/norm(v₀)
            lw = LanczosWorkspace(A, v₀, w.m)
            R = lanczos(lw)
            Tvec = eigvecs(R.T)
            Tval = eigvals(R.T)
            for j in 1:w.m
                τ = Tvec[1, j]
                tr = tr + τ^2 * 1/Tval[j]
            end
            if isapprox(tr, w.result, rtol = w.ctol)
                @show i
                w.nᵥ = i
                break
            end
            w.result = tr
        end
        tr = size(A,1)/w.nᵥ * tr
    else
        println("Matrix is not positive definite")
    end
end
