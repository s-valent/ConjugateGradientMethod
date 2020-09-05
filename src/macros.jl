"""
    @cg [1.0] A \\ b

Solves the least squares problem \$\\min\\|Ax - b\\|^2\$
using conjugate gradient method.

Between @cg and rdiv call `iterpart` can be specified.

## Examples

```julia-repl
julia> @cg [1 2; 3 4] \\ [1, 2]
2-element Array{Float64,1}:
5.551115123125783e-17
0.5000000000000012

julia> @cg 0.5 [1 2; 3 4] \\ [1, 2]
2-element Array{Float64,1}:
0.2343820224719101
0.33483146067415726
```

See also [`cg`](@ref) and [`cg!`](@ref).
"""
macro cg(iterpart::Float64, expr::Expr)
    e = expr

    while isa(e, Expr) && e.head == :call && !isempty(e.args) && first(e.args) != :\
        e = e.args[2]
    end

    if isa(e, Expr) && e.head == :call && length(e.args) == 3 && first(e.args) == :\
        A, b = e.args[2:3]
        e.args = [:cg, esc(A), esc(b), Expr(:kw, :iterpart, iterpart)]
    else
        @warn "Incorrect use of @cg, ignoring..."
    end

    return expr
end

macro cg(expr::Expr)
    e = expr

    while isa(e, Expr) && e.head == :call && !isempty(e.args) && first(e.args) != :\
        e = e.args[2]
    end

    if isa(e, Expr) && e.head == :call && length(e.args) == 3 && first(e.args) == :\
        A, b = e.args[2:3]
        e.args = [:cg, esc(A), esc(b)]
    else
        @warn "Incorrect use of @cg, ignoring..."
    end

    return expr
end
