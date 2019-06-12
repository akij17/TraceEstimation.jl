# Predefined functions and values
invfun(x) = 1/x

function rademacherDistribution!(v)
    o = one(eltype(v))
    v .= Base.rand.(Ref(-o:2*o:o))
end
