"""
Module that defines conjugate gradient method
for non-symmetric matrices.
"""
module GradientMethods

export cg, cg!, @cg

include("common.jl")

include("macros.jl")
include("conjugate-gradient.jl")

end # module
