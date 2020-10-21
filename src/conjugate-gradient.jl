"""
    cg!(x, A, b, [maxiter=size(A,1)])

Solves the least squares problem `min |Ax - b|^2`
using conjugate gradient method. `x` is used as an
initial value and is replaced with the result.

Note that `A` is not symmetric.

## Examples

```julia-repl
julia> x = zeros(2);

julia> cg!(x, [1 2; 3 4], [1, 2])
2-element Array{Float64,1}:
5.551115123125783e-17
0.5000000000000012
```
    
See also [`cg`](@ref) and [`@cg`](@ref).
"""
function cg!(
    x::AbstractVector{T} where T <: Number,
    A::AbstractMatrix{T} where T <: Number,
    b::AbstractVector{T} where T <: Number;
    maxiter::Int = size(A, 1)
)

    checkdims(x, A, b)

    r = similar(x)
    p = similar(x)
    q = similar(x)
    t = similar(b, eltype(x)) 

    mul!(t, A, x)
    @. t -= b
    mul!(r, A', t)
    rr = r ⋅ r
    @. p = r / rr
    mul!(t, A, p)
    mul!(q, A', t)
    pq = p ⋅ q
    @. x -= p / pq

    for _ in 2:maxiter
        @. r -= q / pq
        rr = r ⋅ r
        @. p += r / rr
        mul!(t, A, p)
        mul!(q, A', t)
        pq = p ⋅ q
        @. x -= p / pq
    end

    return x
end


"""
    cg(A, b, [maxiter=size(A,1)])

Solves the least squares problem `min |Ax - b|^2`
using conjugate gradient method.

If you need to use initial value, see `cg!`.

Note that `A` is not symmetric.

## Examples

```julia-repl
julia> cg([1 2; 3 4], [1, 2])
2-element Array{Float64,1}:
5.551115123125783e-17
0.5000000000000012
```
    
See also [`cg!`](@ref) and [`@cg`](@ref).
"""
cg(A, b; args...) = cg!(zerox(A, b), A, b; args...)
