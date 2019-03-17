# Hutchinson Trace Estimator for Inverse of Matrix
# Based on Hutchinson's Stochastic Estimation Proof
# Values from Extrapolation Methods for Estimating Trace of the Matrix Inverse by P. Fika

export hutch

using LinearAlgebra

# Calculating the moments and extrapolating c-1 
function ev(A)
    # Create a vector with +-1 values with 0.5 probability 
    x = rand(-1:2:1, size(A)[1])
    
    # Find c(r) = x' * A^r * x = dot(x', A^r * x)
    c0 = dot(x, x)
    Ax = A * x;
    c1 = dot(x, Ax)
    AAx = A * Ax
    c2 = dot(x, AAx)
    AAAx = A * AAx
    c3 = dot(x, AAAx)

    # Find p (Page 175)
    p = (c0 * c2)/(c1^2)
    
    # Find v0 (Page 179, Numerical Example)  -> Corollary 4 in [16]
    v0 = log10((c1^2)/(c0 * c2)) / log10((c1 * c3)/(c2^2))

    # Finally find ev (Page 175)
    ev = (c0^2)/(p^v0 * c1)
end

#Hutchinson trace estimation using extrapolation technique 
#Page 177
function hutch(A, N=30)
    tr = 0.0
    sum = 0.0
    # (Page 179)
    for i in 1:N
        sum = sum + ev(A)
    end
    tr = sum/N
end

#= TESTS

#Creating a random SPD matrix
n = 5000
A = rand(n, n)
A = A + A' + n*I
#A = [2 -1 0; -1 2 -1; 0 -1 2]

#Running hutchinson
@show @time hutch(A)

#Actual value
@show @time tr(inv(A))

=#
