struct Neuron
    weights::Vector{AbstractNode}
    bias::AbstractNode

    function Neuron(number_inputs::Integer)
        weights = [Node(rand(Distributions.Uniform(-1, 1))) for _ in 1:number_inputs]
        bias = Node(rand(Distributions.Uniform(-1, 1)))
        return new(weights, bias)
    end
end

weights(neuron::Neuron) = getfield(neuron, :weights)
bias(neuron::Neuron) = getfield(neuron, :bias)

function forward!(neuron::Neuron, x::Vector{<:AbstractNode})
    activation = sum(weights(neuron) .* x, init=bias(neuron)) # wx + b
    return tanh(activation)
end




struct Layer
    neurons::Vector{Neuron}

    function Layer(number_inputs::Integer, number_outputs::Integer)
        return new([Neuron(number_inputs) for _ in 1:number_outputs])
    end
end

neurons(layer::Layer) = getfield(layer, :neurons)

function forward!(layer::Layer, x::Vector{<:AbstractNode}) #NOTE: do size checking for x!
    outs = [forward!(neuron, x) for neuron in neurons(layer)]
    return outs
end



struct MLP # Multi-Layer Perceptron
    layers::Vector{Layer}

    function MLP(number_inputs::Integer, number_outputs_list::Vector{<:Integer})
        sizes = vcat([number_inputs], number_outputs_list)
        return new([Layer(sizes[l], sizes[l + 1]) for l in 1:length(number_outputs_list)])
    end
end

layers(mlp::MLP) = getfield(mlp, :layers)

function forward!(mlp::MLP, x::Vector{<:AbstractNode})
    for layer in layers(mlp)
        x = forward!(layer, x)
    end
    return x
end


"""
    Input(x::Vector{<:Real})

Special constructor for defining an input vector of Nodes
"""
function Input(x::Vector{<:Real})
    return Node.(x)
end

struct TrainingSet
    xs::Vector{Vector{LeafNode}} #NOTE: these might always be LeafNodes
    ys::Vector{Float64}

    function TrainingSet(xs::Vector{Vector{<:Real}}, ys::Vector{Real})
        return new(Input.(xs), Float64.(ys))
    end
end