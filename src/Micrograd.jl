module Micrograd

export
    Node,
    Neuron,
    Layer,
    MLP,
    value,
    label,
    label!,
    gradient,
    gradient!,
    backprop!,
    forward!,
    Input,
    DataSet

using
    DataStructures,
    Distributions

include("node.jl")
include("engine.jl")
include("neuralnet.jl")
include("graphviz.jl")

end
