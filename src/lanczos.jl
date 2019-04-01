# Lαnczos iterαtion
# Based on Lanczos method by C. Lanczos

using LinearAlgebra

export LanczosWorkspace, lanczos

mutable struct LanczosWorkspace{T, TM <:AbstractMatrix{T}, TN <: AbstractMatrix{T}, TV <:AbstractVector{T}, I <:Integer}
    A::TM
    V::TN
    T::TN
    v::TV
    vₒ::TV
    ω::TV
    α::T
    β::T
    m::I
end

struct LanczosResult
    V::AbstractMatrix
    T::AbstractMatrix
end
"""
    LanczosWorkspace(A, v, m)

Create a Lanczos Workspace struct to use with Lanczos iteration.
# Arguments
    - `A` : Symmetric Positive Definite Matrix
    - `v` : An arbitary vector for a function: f -> A*v
    - `m` : Number of iterations
"""
function LanczosWorkspace(A, v, m)
    n = size(v, 1)
    m = m > n ? n : m
    V = zeros(eltype(A), n, m)
    T = zeros(eltype(A), m, m)
    vₒ = zeros(eltype(A), n)
    β = zero(eltype(A))
    ω = similar(v)
    α = zero(eltype(A))
    return LanczosWorkspace(A, V, T, v, vₒ, ω, α, β, m)
end

"""
    lanczos(lw::LanczosWorkspace)

Takes a LanczosWorkspace object and returns LanczosResult object.
Access SymTridiagonal T by R.T and orthonormal matrix V by R.V

Example:
lw = LanczosWorkspace(A, v, m)
R = lanczos(lw)
"""
function lanczos(lw::LanczosWorkspace)
    for i in 1:lw.m
        mul!(lw.ω, lw.A, lw.v)
        lw.α = dot(lw.ω, lw.v)
        lw.ω = lw.ω - (lw.α * lw.v) - (lw.β * lw.vₒ)
        lw.β = norm(lw.ω)
        lw.T[i, i] = lw.α
        if i < lw.m
            lw.T[i, i+1] = lw.β
            lw.T[i+1, i] = lw.β
        end
        lw.vₒ= lw.v
        lw.v = lw.ω/lw.β
        lw.V[:, i] = lw.v
    end
    LanczosResult(lw.V, lw.T)
end
