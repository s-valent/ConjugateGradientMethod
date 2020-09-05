import LinearAlgebra

"""
    ConjugateGradientMethod.checkdims(x, A, b)

checks if arguments have correct dimensions for cg method.
"""
@inline function checkdims(X, A, B)
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
    ConjugateGradientMethod.zerox(length)

used in `cg` function (not `cg!`).
"""
@inline function zerox(A, b)
    T = typeof(zero(eltype(b)) / one(eltype(A)))
    return zeros(T, size(A, 2))
end

"""
    ConjugateGradientMethod.dot(x, y) = LinearAlgebra.dot(x, y)

dot product function, used in cg method.
"""
@inline dot(x, y) = LinearAlgebra.dot(x, y)

const ⋅ = dot

"""
    ConjugateGradientMethod.mul!(y, A, x) = LinearAlgebra.mul!(y, A, x)

matrix-vector multilpication function, used in cg method.
"""
@inline mul!(y, A, x) = LinearAlgebra.mul!(y, A, x)

"""
    ConjugateGradientMethod.close!(x) = nothing

function that is called after arrays inside cg method are 
no longer needed.

can be redefined by user to free memory.
"""
@inline close!(x) = nothing
