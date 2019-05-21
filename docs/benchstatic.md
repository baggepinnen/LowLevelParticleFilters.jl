In this script, we'll do some benchmarking to determine how to do the dynamics update. Should we do it inplace, using static arrays etc.?

````julia
using StaticArrays, BenchmarkTools, Statistics

N = 100 # Number of particles
d = 4   # Number of dimensions
x = [zeros(d) for _ in 1:N]
x2 = [zeros(d) for _ in 1:N]
xs = [@SVector(zeros(d)) for _ in 1:N]
xs2 = [@SVector(zeros(d)) for _ in 1:N]

f = x->x.^2 # Simple dynamics functions
g = x->x^2  # Simple dynamics functions
````


````
#11 (generic function with 1 method)
````





below are four different ways of calling the dynamics

````julia
function f_inplace!(x2,x,f)
    for i = eachindex(x)
        x2[i] .= f(x[i])
    end
end

function f_inplace_b!(x2,x,f)
    for i = eachindex(x)
        x2[i] .= f.(x[i])
    end
end

function f_update!(x2,x,f)
    for i = eachindex(x)
        x2[i] = f(x[i])
    end
end

function f_update_b!(x2,x,f)
    for i = eachindex(x)
        x2[i] = f.(x[i])
    end
end;
````





Now we do some benchmarking. We do `evals=1000` to simulate the performance for 1000 time steps, including potential gc time etc.

````julia
b1 = @benchmark f_inplace!($x2,$x,$f) evals=1000
b2 = @benchmark f_inplace_b!($x2,$x,$g) evals=1000
b3 = @benchmark f_update!($x2,$x,$f) evals=1000
b4 = @benchmark f_update!($xs2,$xs,$f) evals=1000
b5 = @benchmark f_update_b!($x2,$x,$g) evals=1000
b6 = @benchmark f_update_b!($xs2,$xs,$g) evals=1000

bs = [b1,b2,b3,b4,b5,b6]
display([ratio(mean(b1),mean(b2)) for b1 in bs, b2 in bs])
````


````
6×6 Array{BenchmarkTools.TrialRatio,2}:
 100.00%  55.37%   115.59%  6889.80%   34.48%   7065.72%
 180.59%  100.00%  208.74%  12442.46%  62.27%   12760.16%
 86.51%   47.91%   100.00%  5960.63%   29.83%   6112.82%
 1.45%    0.80%    1.68%    100.00%    0.50%    102.55%  
 290.03%  160.60%  335.24%  19982.40%  100.00%  20492.60%
 1.42%    0.78%    1.64%    97.51%     0.49%    100.00%
````



````julia
display([ratio(memory(b1),memory(b2)) for b1 in bs, b2 in bs])
````


````
6×6 Array{Float64,2}:
 1.0       3.5  1.0       Inf    0.777778  Inf  
 0.285714  1.0  0.285714  Inf    0.222222  Inf  
 1.0       3.5  1.0       Inf    0.777778  Inf  
 0.0       0.0  0.0         1.0  0.0         1.0
 1.28571   4.5  1.28571   Inf    1.0       Inf  
 0.0       0.0  0.0         1.0  0.0         1.0
````





We now add the `::F) where F` to force specialization on `f`

````julia
function f_inplace!(x2,x,f::F) where F
    for i = eachindex(x)
        x2[i] .= f(x[i])
    end
end

function f_inplace_b!(x2,x,f::F) where F
    for i = eachindex(x)
        x2[i] .= f.(x[i])
    end
end

function f_update!(x2,x,f::F) where F
    for i = eachindex(x)
        x2[i] = f(x[i])
    end
end

function f_update_b!(x2,x,f::F) where F
    for i = eachindex(x)
        x2[i] = f.(x[i])
    end
end;
````





Now we do some benchmarking. We do `evals=1000` to simulate the performance for 1000 time steps, including potential gc time etc.

````julia
b7 = @benchmark f_inplace!($x2,$x,$f) evals=1000
b8 = @benchmark f_inplace_b!($x2,$x,$g) evals=1000
b9 = @benchmark f_update!($x2,$x,$f) evals=1000
b10 = @benchmark f_update!($xs2,$xs,$f) evals=1000
b11 = @benchmark f_update_b!($x2,$x,$g) evals=1000
b12 = @benchmark f_update_b!($xs2,$xs,$g) evals=1000

bs2 = [b7, b8, b9, b10, b11, b12]
display(ratio.(mean.(bs), mean.(bs2)))
````


````
6-element Array{BenchmarkTools.TrialRatio,1}:
 105.41%
 1866.31%
 101.65%
 112.88%
 362.32%
 109.16%
````



````julia
display(ratio.(memory.(bs), memory.(bs2)))
````


````
6-element Array{Float64,1}:
   1.0               
 Inf                 
   1.0               
   1.0               
   1.2857142857142858
   1.0
````





NOTE: It was absolutely crucial to do the `f::F) where F`. Without this, b6 was much slower than b4, and b2 and b5 were really slow

````julia
display([ratio(mean(b1),mean(b2)) for b1 in bs2, b2 in bs2])
````


````
6×6 Array{BenchmarkTools.TrialRatio,2}:
 100.00%  980.41%  111.47%  7378.07%  118.52%  7317.06%
 10.20%   100.00%  11.37%   752.55%   12.09%   746.32%
 89.71%   879.51%  100.00%  6618.74%  106.32%  6564.01%
 1.36%    13.29%   1.51%    100.00%   1.61%    99.17%  
 84.38%   827.23%  94.06%   6225.32%  100.00%  6173.84%
 1.37%    13.40%   1.52%    100.83%   1.62%    100.00%
````



````julia
display([ratio(memory(b1),memory(b2)) for b1 in bs2, b2 in bs2])
````


````
6×6 Array{Float64,2}:
 1.0  Inf    1.0  Inf    1.0  Inf  
 0.0    1.0  0.0    1.0  0.0    1.0
 1.0  Inf    1.0  Inf    1.0  Inf  
 0.0    1.0  0.0    1.0  0.0    1.0
 1.0  Inf    1.0  Inf    1.0  Inf  
 0.0    1.0  0.0    1.0  0.0    1.0
````





It seems like a dynamics function which can return a StaticVector (f_update!) is 7 times faster than inplace modification of a regular array (f_inplace_b!)

Just for fun, let's also try inplace for MVectors

````julia
xm = [@MVector(zeros(d)) for _ in 1:N]
xm2 = [@MVector(zeros(d)) for _ in 1:N]
b13 = @benchmark f_inplace_b!($xm2,$xm,$g) evals=1000
bs3 = [b2,b4,b13]
display([ratio(mean(b1),mean(b2)) for b1 in bs3, b2 in bs3])
````


````
3×3 Array{BenchmarkTools.TrialRatio,2}:
 100.00%  12442.46%  4154.38%
 0.80%    100.00%    33.39%  
 2.41%    299.50%    100.00%
````



````julia
display([ratio(memory(b1),memory(b2)) for b1 in bs3, b2 in bs3])
````


````
3×3 Array{Float64,2}:
 1.0  Inf    Inf  
 0.0    1.0    1.0
 0.0    1.0    1.0
````





It seems we pay a factor of 2-3 for the MMatrix

We now consider statically sized regular vectors

````julia
xv = [SizedVector{d}(zeros(d)) for _ in 1:N]
xv2 = [SizedVector{d}(zeros(d)) for _ in 1:N]
b14 = @benchmark f_inplace_b!($xv2,$xv,$g) evals=1000
bs4 = [b2,b4,b14]
display([ratio(mean(b1),mean(b2)) for b1 in bs4, b2 in bs4])
````


````
3×3 Array{BenchmarkTools.TrialRatio,2}:
 100.00%  12442.46%  2471.98%
 0.80%    100.00%    19.87%  
 4.05%    503.34%    100.00%
````



````julia
display([ratio(memory(b1),memory(b2)) for b1 in bs4, b2 in bs4])
````


````
3×3 Array{Float64,2}:
 1.0  Inf    Inf  
 0.0    1.0    1.0
 0.0    1.0    1.0
````





It seems we pay a factor of 5 for the SizedVector

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
