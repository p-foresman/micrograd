module Micrograd

export
    Value,
    draw_dot

using
    Revise

include("engine.jl")
include("graphviz.jl")

end # module Micrograd
