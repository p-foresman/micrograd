module Micrograd

export
    Node,
    value,
    label,
    label!,
    gradient,
    gradient!,
    backprop,
    draw_dot

using DataStructures


# include("operations.jl")
include("node_working.jl")
# include("node.jl")
# include("engine.jl")
# include("graphviz.jl")

end # module Micrograd
