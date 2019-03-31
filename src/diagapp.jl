# Diagonal Approximation for trace of inverse of matrix
# Values from Extrapolation Methods for Estimating Trace of the Matrix Inverse by P. Fika

export diagapp

using LinearAlgebra

struct Diagappspace{T, TM <: AbstractMatrix{T}, TV <: AbstractVector{T}, F<:Function}
    A::TM
    x::TV
    y::TV
    randfunc::F
end

function Diagappspace(A::AbstractMatrix, randfunc::Function)
    x = randfunc(eltype(A) <: Integer ? Float64 : eltype(A), size(A, 1))
    y = similar(x)
    return Diagappspace(A, x, y, randfunc)
end

# Extrapolation for c-1 and calculating value of v0
function v0(w::Diagappspace)

    copyto!(w.x, w.randfunc(-1:2:1, size(w.A, 1)))

    c0 = dot(w.x, w.x)
    mul!(w.y, w.A, w.x)
    c1 = dot(w.y, w.x)
    c2 = dot(w.y, w.y)
    mul!(w.x, w.A, w.y)
    c3 = dot(w.y, w.x)    

    p = (c0 * c2)/(c1^2)
    
    v0 = log10((c1^2)/(c0 * c2)) / log10((c1 * c3)/(c2^2))
end

# Calculating the Approximation for ith value of inverse diagonal
function dfun(A, i, v)
    isum = zero(eltype(A))
    s = size(A, 1)
    for k in 1:s
        isum = isum + A[k, i]^2
    end
    #v = v0(A)
    p = isum / (A[i, i]^2)
    d = 1 / (p^v * A[i, i])
end

# Calculating Diagonal Approximation for Matrix Inverse
# This works a SPD Matrix with low condition number 
"""
    diagapp(A::AbstractMatrix; randfunc=Base.rand)

Diagonal Approximation algorithm for inverse of matrix.
# Arguments
 - `A` : Symmetric Positive Definite Matrix with Low Condtion Number (k < 500)
 - `randfunc` : Random function for the particular type of matrix A
                (An example can be found in test/diagapp.jl)
"""
function diagapp(A; randfunc::Function=Base.rand)
    tr = zero(eltype(A))
    s = size(A, 1)
    w = Diagappspace(A, randfunc)
    v = v0(w)
    for i in 1:s
        tr = tr + dfun(w.A, i, v)
    end
    tr
end
