# Hutchinson Trace Estimator for Inverse of Matrix
# Based on Hutchinson's Stochastic Estimation Proof
# Values from Extrapolation Methods for Estimating Trace of the Matrix Inverse by P. Fika

export hutch, hutch!, HutchWorkspace

using LinearAlgebra
using CuArrays

struct HutchWorkspace
    A::AbstractArray{Float64, 2}
    x::AbstractArray{Float64, 1}
    y::AbstractArray{Float64, 1}
    N::Int64
    skipverify::Bool
end

"""
    HutchWorkspace(A)

# Arguments
 - `A` : Symmetric Hermitian Matrix 
 - `N` : Number of iterations (Default: 30)
 - `skipverify` : If false, it will check isposdef(A) (Default: false)
"""
function HutchWorkspace(A; N = 30, skipverify = false)
    x = rand(-1:2:1, size(A)[1])
    y = similar(x)
    return HutchWorkspace(A, x, y, N, skipverify)
end


# Calculating the moments and extrapolating c-1 
function ev(w::HutchWorkspace)
    # Create new random values for x
    copyto!(w.x, rand(-1:2:1, size(w.A)[1]))

    # Find c(r) = x' * A^r * x = dot(x', A^r * x)
    c0 = dot(w.x, w.x)
    mul!(w.y, w.A, w.x)
    c1 = dot(w.y, w.x)
    c2 = dot(w.y, w.y)
    mul!(w.x, w.A, w.y)
    c3 = dot(w.y, w.x)

    # Find p (Page 175)
    p = (c0 * c2)/(c1^2)
    
    # Find v0 (Page 179, Numerical Example) -> Corollary 4 in [16]
    v0 = log10((c1^2)/(c0 * c2)) / log10((c1 * c3)/(c2^2))

    # Finally find ev (Page 175)
    ev = (c0^2)/(p^v0 * c1)
end
"""
    hutch(A; N = 30, skipverify = false)

Take a HutchWorkspace object as input, apply hutchinson estimation algorithm and solve for trace 
of inverse of the matrix. (in-place version also available)
"""
# Hutchinson trace estimation using extrapolation technique (Page 177)
function hutch(A; N=30, skipverify = false)
    w = HutchWorkspace(A, N = N, skipverify = skipverify)
    return hutch!(w)
end
"""
    hutch!(w::HutchWorkspace)

Take a HutchWorkspace object as input, apply hutchinson estimation algorithm and solve for trace 
of inverse of the matrix. (in-place version of hutch)
"""
function hutch!(w::HutchWorkspace)
    if w.skipverify == true || isposdef(w.A)
        tr = 0.0
        sum = 0.0
        for i in 1:w.N
            sum = sum + ev(w)
        end
        tr = sum/w.N
    end
end

#=
A = rand(5000, 5000)
for i in 1:5000
    for j in 1:5000
        A[i, j] = exp(-2 * abs(i - j))
    end
end
w = HutchWorkspace(A, skipverify = false)
@time @show hutch(A)
@time @show hutch!(w)
@time @show tr(inv(A))
=#