using LinearAlgebra

"""
    GradientMethods.checkdims(x, A, b)

checks if arguments have correct dimensions for cg method.
"""
function checkdims(X, A, B)
    nA, mA = size(A)
    nB = length(B)
    nX = length(X)

    if nA != nB
        throw(DimensionMismatch("matrix A has dimensions ($nA,$mA)), vector B has length $nB"))
    end

    if mA != nX
        throw(DimensionMismatch("result x has length $nX, needs length $mA"))
    end
end

"""
    GradientMethods.zerox(A, b)

used in `cg` function (not `cg!`).
"""
function zerox(A, b)
    T = typeof(zero(eltype(b)) / one(eltype(A)))
    return zeros(T, size(A, 2))
end
