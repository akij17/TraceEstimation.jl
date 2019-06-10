# Stochastic Chebyshev Polynomial method
# Based on APPROXIMATING SPECTRAL SUMS OF LARGE-SCALE MATRICES USING STOCHASTIC CHEBYSHEV APPROXIMATIONS By Han, Malioutov, Avron and Shin

using LinearAlgebra

𝓍(k, n) = cos((π * (k + 0.5))/(n+1))
const Tvals = Dict{Int, Float64}()
fn(x) = 1/(x)

function T(j, x)
    if j == 0
        return 1
    end
    if j == 1
        return x
    end
    get!(Tvals, j) do
        return 2*x*T(j-1, x) - T(j-2, x)
    end
end

function coeff(j, n, a, b)
    functionsum = 0.0
    for k in 0:n
        x = 𝓍(k, n)
        functionsum = fn((((b-a)/2) * x) + (b+a)/2) * T(j, x)
    end
    if j == 0
        return (1/(n+1)) * functionsum
    end
    return (2/(n+1)) * functionsum
end

function rademacherDistribution(n, T::Type)
    o = one(T)
    v = Base.rand(-o:2*o:o, n)
end

function chebyshev(A, a, b, m, n)
    tr = zero(eltype(A))
    C = []
    for j in 0:n
        @show cⱼ = coeff(j, n, a, b)
        push!(C, cⱼ)
    end
    s = size(A, 1)
    w₀ = zeros(eltype(A), s)
    w₁ = zeros(eltype(A), s)
    w₂ = zeros(eltype(A), s)
    u = zeros(eltype(A), s)
    for i in 1:m
        v = rademacherDistribution(s, eltype(A))
        w₀ .= v
        mul!(w₁, (2/(b-a)) * A, v)
        w₁ .= w₁ .- (((b+a)/(b-a)) .* v)
        u .= C[1]*w₀ .+ C[2]*w₁
        for j in 2:n
            mul!(w₂, (4/(b-a)) * A, w₁)
            w₂ .= w₂ .- ((2(b+a)/(b-a)) .* w₁) .- w₀
            u .= u .+ C[j+1]*w₂
            w₀ .= w₁
            w₁ .= w₂
        end
        @show tr = tr + (v' * u)/m
    end
    return tr
end

A = rand(2234,2234)
A = A + A'
while !isposdef(A)
    global A = A + 10I
end

a = eigmin(A) - 0.5
b = eigmax(A) + 0.5

@time @show chebyshev(A, a, b, 4,6)
@time @show tr(inv(A))
