# In this script, we'll do some benchmarking to determine how to do the dynamics update. Should we do it inplace, using static arrays etc.?
using StaticArrays, BenchmarkTools, Statistics

N = 100 # Number of particles
d = 4   # Number of dimensions
x = [zeros(d) for _ in 1:N]
x2 = [zeros(d) for _ in 1:N]
xs = [@SVector(zeros(d)) for _ in 1:N]
xs2 = [@SVector(zeros(d)) for _ in 1:N]

f = x->x.^2 # Simple dynamics functions
g = x->x^2  # Simple dynamics functions

# below are four different ways of calling the dynamics
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

# Now we do some benchmarking. We do `evals=1000` to simulate the performance for 1000 time steps, including potential gc time etc.
b1 = @benchmark f_inplace!($x2,$x,$f) evals=1000
b2 = @benchmark f_inplace_b!($x2,$x,$g) evals=1000
b3 = @benchmark f_update!($x2,$x,$f) evals=1000
b4 = @benchmark f_update!($xs2,$xs,$f) evals=1000
b5 = @benchmark f_update_b!($x2,$x,$g) evals=1000
b6 = @benchmark f_update_b!($xs2,$xs,$g) evals=1000

bs = [b1,b2,b3,b4,b5,b6]
display([ratio(mean(b1),mean(b2)) for b1 in bs, b2 in bs])
display([ratio(memory(b1),memory(b2)) for b1 in bs, b2 in bs])


# We now add the `::F) where F` to force specialization on `f`
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

# Now we do some benchmarking. We do `evals=1000` to simulate the performance for 1000 time steps, including potential gc time etc.
b7 = @benchmark f_inplace!($x2,$x,$f) evals=1000
b8 = @benchmark f_inplace_b!($x2,$x,$g) evals=1000
b9 = @benchmark f_update!($x2,$x,$f) evals=1000
b10 = @benchmark f_update!($xs2,$xs,$f) evals=1000
b11 = @benchmark f_update_b!($x2,$x,$g) evals=1000
b12 = @benchmark f_update_b!($xs2,$xs,$g) evals=1000

bs2 = [b7, b8, b9, b10, b11, b12]
display(ratio.(mean.(bs), mean.(bs2)))
display(ratio.(memory.(bs), memory.(bs2)))
# NOTE: It was absolutely crucial to do the `f::F) where F`. Without this, b6 was much slower than b4, and b2 and b5 were really slow

display([ratio(mean(b1),mean(b2)) for b1 in bs2, b2 in bs2])
display([ratio(memory(b1),memory(b2)) for b1 in bs2, b2 in bs2])
# It seems like a dynamics function which can return a StaticVector (f_update!) is 7 times faster than inplace modification of a regular array (f_inplace_b!)

# Just for fun, let's also try inplace for MVectors
xm = [@MVector(zeros(d)) for _ in 1:N]
xm2 = [@MVector(zeros(d)) for _ in 1:N]
b13 = @benchmark f_inplace_b!($xm2,$xm,$g) evals=1000
bs3 = [b2,b4,b13]
display([ratio(mean(b1),mean(b2)) for b1 in bs3, b2 in bs3])
display([ratio(memory(b1),memory(b2)) for b1 in bs3, b2 in bs3])
# It seems we pay a factor of 2-3 for the MMatrix

# We now consider statically sized regular vectors
xv = [SizedVector{d}(zeros(d)) for _ in 1:N]
xv2 = [SizedVector{d}(zeros(d)) for _ in 1:N]
b14 = @benchmark f_inplace_b!($xv2,$xv,$g) evals=1000
bs4 = [b2,b4,b14]
display([ratio(mean(b1),mean(b2)) for b1 in bs4, b2 in bs4])
display([ratio(memory(b1),memory(b2)) for b1 in bs4, b2 in bs4])
# It seems we pay a factor of 5 for the SizedVector

#src Compile using literateweave("benchstatic.jl", doctype="github")
