include("custom-usage.jl")


function __iternum(default::Int, args)::Int
    if length(args) == 0
        return default
    elseif length(args) == 1
        if :iternum in keys(args)
            iternum = args[:iternum]
            try return iternum::Int catch end
        elseif :iterpart in keys(args)
            iterpart = args[:iterpart]
            try return ceil(Int, default * Float64(iterpart)) catch end
        end
    end

    throw(ArgumentError("Incorrect arguments for cg method"))
end


"""
    cg!(x, A, b, [iterpart=1.0])
    cg!(x, A, b, [iternum=size(A,1)])

Solves the least squares problem \$\\min\\|Ax - b\\|^2\$
using conjugate gradient method. `x` is used as an initial value
and is replaced with the result.

`iternum` specifies the number of iterations to stop after
(minimum 1).

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
    b::Vector{T} where T <: Number;
    args...
)

    checkdims(x, A, b)
    iternum = __iternum(size(A, 1), args)

    r = similar(x)
    p = similar(x)
    q = similar(x)
    t = similar(b, eltype(x))

    mul!(t, A, x)
    @. t -= b
    mul!(r, transpose(A), t)
    rr = r ⋅ r
    @. p = r / rr
    mul!(t, A, p)
    mul!(q, transpose(A), t)
    pq = p ⋅ q
    @. x -= p / pq

    for _ in 2:iternum
        @. r -= q / pq
        rr = r ⋅ r
        @. p += r / rr
        mul!(t, A, p)
        mul!(q, transpose(A), t)
        pq = p ⋅ q
        @. x -= p / pq
    end

    close!(r)
    close!(p)
    close!(q)
    close!(t)

    return x
end


"""
    cg(A, b, [iterpart=1.0])
    cg(A, b, [iternum=size(A,1)])

Solves the least squares problem \$\\min\\|Ax - b\\|^2\$
using conjugate gradient method. To use initial value see `cg!`.

`iternum` specifies the number of iterations to stop after
(minimum 1). `iterpart` = `iternum` / ``size(A,1)``

## Examples

```julia-repl
julia> cg([1 2; 3 4], [1, 2])
2-element Array{Float64,1}:
5.551115123125783e-17
0.5000000000000012
```
    
See also [`cg!`](@ref) and [`@cg`](@ref).
"""
@inline function cg(
    A::AbstractMatrix{T} where T <: Number,
    b::Vector{T} where T <: Number;
    args...
)

    x = zerox(A, b)
    return cg!(x, A, b; args...)
end
