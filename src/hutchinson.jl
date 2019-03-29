# Hutchinson Trace Estimator for Inverse of Matrix
# Based on Hutchinson's Stochastic Estimation Proof
# Values from Extrapolation Methods for Estimating Trace of the Matrix Inverse by P. Fika
# (This method works for SPD with low condition number)

export hutch, hutch!, HutchWorkspace

using LinearAlgebra

struct HutchWorkspace{T, TM <: AbstractMatrix{T}, F <: Function, TV <: AbstractVector{T}}
    A::TM
    randfunc::F
    x::TV
    y::TV
    N::Int64
    skipverify::Bool
end

"""
    HutchWorkspace(A; N = 30, skipverify = false)
    HutchWorkspace(A, randfunc::Function; N = 30, skipverify = false)

# Arguments
- `A` : Symmetric Positive Definite Matrix with Low Condtion Number (k < 500)
- `randfunc` : Function to generate random values for x distributed uniformly 
                (Base: rand(-1.0:2.0:1.0, size(A,1))
                 Example: f(n) = rand(-1.0:2.0:1.0, n)
- `N` : Number of iterations (Default: 30)
- `skipverify` : If false, it will check isposdef(A) (Default: false)
"""
function HutchWorkspace(A, randfunc::Function; N = 30, skipverify = false)
    x = randfunc(size(A,1))
    y = similar(x)
    return HutchWorkspace(A, randfunc, x, y, N, skipverify)
end

function HutchWorkspace(A; N = 30, skipverify = false)
    randfunc(n) = rand(-1:2:1, n)
    x = rand(eltype(A) <: Integer ? Float64 : eltype(A), size(A,1))
    y = similar(x)
    return HutchWorkspace(A, randfunc, x, y, N, skipverify)
end

# Calculating the moments and extrapolating c-1 
function ev(w::HutchWorkspace)
    # Create new random values for x
    copyto!(w.x, w.randfunc(size(w.A, 1)))

    # Find c(r) = x' * A^r * x = dot(x', A^r * x)
    c0 = dot(w.x, w.x)
    mul!(w.y, w.A, w.x)
    c1 = dot(w.y, w.x)
    c2 = dot(w.y, w.y)
    mul!(w.x, w.A, w.y)
    c3 = dot(w.y, w.x)

    # intermediate arguments for v0 
    nVal = (c1^2)/(c0 * c2)
    dVal = (c1 * c3)/(c2^2)

    if (nVal > 0) && (dVal > 0)
        # Find p (Page 175)
        p = (c0 * c2)/(c1^2)
        # Find v0 (Page 179, Numerical Example) -> Corollary 4 in [16]
        v0 = log10(nVal) / log10(dVal)
        # Finally find ev (Page 175)
        ev0 = (c0^2)/(p^v0 * c1)
        return ev0
    else
        @warn "v0 cannot be calculated. Aitken's process may give inaccurate results!"
        w1 = HutchWorkspace(w.A, w.randfunc, w.x, w.y, w.N, w.skipverify)
        return hutch!(w1, aitken=true)
    end
end

# Aitken's Process to Predict the negative moment
# (Page 176, eq 4)        
function gfun(w::HutchWorkspace)
    copyto!(w.x, w.randfunc(size(w.A, 1)))

    c0 = dot(w.x, w.x)
    mul!(w.y, w.A, w.x)
    c1 = dot(w.y, w.x)
    c2 = dot(w.y, w.y)
    mul!(w.x, w.A, w.y)
    c3 = dot(w.y, w.x)

    #g1 = c0 - (((c1 - c0)^2) / (c2 - c1))
    g2 = c1 - (((c2 - c0) * (c2 - c1)) / (c3 - c2))
    #g = (g1 + g2)/2
end

"""
    hutch(A; N = 30, skipverify = false)

Take a HutchWorkspace object as input, apply hutchinson estimation algorithm and solve for trace 
of inverse of the matrix. (in-place version also available)
"""
# Hutchinson trace estimation using extrapolation technique (Page 177)
function hutch(A; N=30, skipverify = false, aitken = false)
    w = HutchWorkspace(A, N = N, skipverify = skipverify)
    return hutch!(w, aitken)
end

"""
    hutch!(w::HutchWorkspace; aitken = false)
    hutch!(A::AbstractArray; N = 30, skipverify = false, aitken = false)

Take a HutchWorkspace object as input, apply hutchinson estimation algorithm and solve for trace 
of inverse of the matrix. (in-place version of hutch)
"""
function hutch!(w::HutchWorkspace; aitken = false)
    if w.skipverify == true || isposdef(w.A)
        tr = zero(eltype(w.A))
        sum = zero(eltype(w.A))
        for i in 1:w.N
            # if aitken == true => use aitken process to predict the terms via gfun
            sum = sum + (aitken ? gfun(w) : ev(w))
        end
        tr = sum/w.N
    end
end

function hutch!(A::AbstractArray{<:Any, 2}; N = 30, skipverify = false, aitken = false)
    w = HutchWorkspace(A, N, skipverify)
    return hutch!(w, aitken)
end
