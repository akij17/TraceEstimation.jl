# Diagonal Approximation for trace of inverse of matrix
# Values from Extrapolation Methods for Estimating Trace of the Matrix Inverse by P. Fika

export diagapp, diagapp!

using LinearAlgebra

struct diagappspace
    A::AbstractArray{<:Any, 2}
    x::AbstractArray{<:Any, 1}
    y::AbstractArray{<:Any, 1}
end

function diagappspace(A::AbstractArray{<:Any, 2})
    x = rand(-1.0:2.0:1.0, size(A)[1])
    y = similar(x, Float64)
    return diagappspace(A, x, y)
end

# Extrapolation for c-1 and calculating value of v0
function v0(w::diagappspace)

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
    isum = 0.0
    s = size(A)[1]
    for k in 1:s
        isum = isum + A[k, i]^2
    end
    #v = v0(A)
    p = isum / (A[i, i]^2)
    d = 1 / (p^v * A[i, i])
end

# Calculating Diagonal Approximation for Inverse of Matrix
"""
    diagapp(A::AbstractArray)

Diagonal Approximation algorithm for inverse of matrix (SPD low condition number).
"""
function diagapp(A)
    return diagapp!(A) 
end

"""
    diagapp!(A::AbstractArray)

Diagonal Approximation algorithm for inverse of matrix (SPD low condition number).
"""
function diagapp!(A)
    tr = 0.0
    s = size(A)[1]
    w = diagappspace(A)
    v = v0(w)
    for i in 1:s
        tr = tr + dfun(w.A, i, v)
    end
    tr
end
