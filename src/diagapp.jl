# Diagonal Approximation for trace of inverse of matrix
# Based on moment prediction by Aitken's Process
# As described in Extrapolation Methods for Estimating Trace of the Matrix Inverse by P. Fika

export diagapp

using LinearAlgebra

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

function gfun(w)
    
    # Create new random values for x
    copyto!(w.x, rand(-1:2:1, size(w.A)[1]))

    # Find c(r) = x' * A^r * x = dot(x', A^r * x)
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

function diagapp(w::HutchWorkspace)
    if w.skipverify == true || isposdef(w.A)
        tr = 0.0
        sum = 0.0
        for i in 1:w.N
            sum = sum + gfun(w)
        end
        tr = sum/w.N
    end
end

A = rand(5000, 5000)

A = A + A' + (5000 * I)

w = HutchWorkspace(A, skipverify = true)
@time @show diagapp(w)
@time @show tr(inv(A))