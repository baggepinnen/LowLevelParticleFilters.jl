"""
    SignalNames(; x, u, y, name)

A structure representing the names of the signals in a system.

- `x::Vector{String}`: Names of the state variables
- `u::Vector{String}`: Names of the input variables
- `y::Vector{String}`: Names of the output variables
- `name::String`: Name of the system
"""
@kwdef struct SignalNames
    x::Vector{String} = [""]
    u::Vector{String} = [""]
    y::Vector{String} = [""]
    name::String
end

function default_names(nx, nu, ny, name="")
    x = ["x$i" for i in 1:nx]
    u = ["u$i" for i in 1:nu]
    y = ["y$i" for i in 1:ny]
    SignalNames(x, u, y, string(name))
end

