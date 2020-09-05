### ConjugateGradientMethod.jl

A basic implementation of conjugate gradient method for non-symmetric matrices.

Usage:
```julia-repl
julia> cg([1 2; 3 4], [1, 2])
2-element Array{Float64,1}:
5.551115123125783e-17
0.5000000000000012

julia> @cg [1 2; 3 4] \\ [1, 2]
2-element Array{Float64,1}:
5.551115123125783e-17
0.5000000000000012
```


Conjugate Gradient Method is great because it can be stopped without doing all number of iterations.
```julia-repl
julia> using ConjugateGradientMethod

julia> using BenchmarkTools

julia> const A = rand(3000, 2500);

julia> const x = rand(2500);

julia> const b = A * x;

julia> function test()
           r1 = A \ b
           r2 = @cg A \ b
           r3 = @cg 0.1 A \ b
           
           @btime A \ b
           @btime @cg A \ b
           @btime @cg 0.1 A \ b
           
           println(maximum(abs.(r1-x)))
           println(maximum(abs.(r2-x)))
           println(maximum(abs.(r3-x)))
       end; test()
  3.562 s (15051 allocations: 154.94 MiB)
  16.535 s (10 allocations: 102.08 KiB)
  1.784 s (10 allocations: 102.08 KiB)
3.8824499171141724e-13
6.782163719520895e-11
9.90383608456824e-11
```

