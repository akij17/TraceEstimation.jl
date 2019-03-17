# Diagonal Approximation for trace of inverse of matrix

export diagapp

using LinearAlgebra

# Extrapolation for c-1 and calculating value of v0
function v0(A)
    x = rand(-1:2:1, size(A)[1])
    
    c0 = dot(x, x)
    Ax = A * x;
    c1 = dot(x, Ax)
    AAx = A * Ax
    c2 = dot(x, AAx)
    AAAx = A * AAx
    c3 = dot(x, AAAx)

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
function diagapp(A)
    tr = 0.0
    s = size(A)[1]
    v = v0(A)
    for i in 1:s
       tr = tr + dfun(A, i, v)
    end
    tr 
end

#= TESTS
#Creating a random SPD matrix
n = 10000
A = rand(n, n)
A = A + A' + n*I
#A = [2 -1 0; -1 2 -1; 0 -1 2]

#Running hutchinson
@show @time diagapp(A)

#Actual value
@show @time tr(inv(A))
=#
