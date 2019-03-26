# Diagonal Approximation for trace of inverse of matrix

export diagapp

using LinearAlgebra

# Extrapolation for c-1 and calculating value of v0
function v0(A)
    x = rand(-1.0:2.0:1.0, size(A)[1])
    y = similar(x, Float64)
    
    c0 = dot(x, x)
    mul!(y, A, x)
    c1 = dot(y, x)
    c2 = dot(y, y)
    mul!(x, A, y)
    c3 = dot(y, x)    

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
    diagapp(A)

Diagonal Approximation algorithm for inverse of matrix.
"""
function diagapp(A)
    tr = 0.0
    s = size(A)[1]
    v = v0(A)
    for i in 1:s
       tr = tr + dfun(A, i, v)
    end
    tr 
end
