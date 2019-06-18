# Predefined functions and values
const ϵ = 0.5
const ξ = 0.01

invfun(x) = 1/x

function rademacherDistribution!(v)
    o = one(eltype(v))
    v .= Base.rand.(Ref(-o:2*o:o))
end
