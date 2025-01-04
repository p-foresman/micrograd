struct Neuron
    weights::Vector{AbstractNode}
    bias::AbstractNode
    # squash::Function #Symbol?

    function Neuron(number_inputs::Integer)
        weights = [Node(rand(Distributions.Uniform(-1, 1))) for _ in 1:number_inputs]
        bias = Node(rand(Distributions.Uniform(-1, 1)))
        return new(weights, bias)
    end
end

weights(neuron::Neuron) = getfield(neuron, :weights)
bias(neuron::Neuron) = getfield(neuron, :bias)
parameters(neuron::Neuron) = vcat(weights(neuron), [bias(neuron)])

function forward!(neuron::Neuron, x::Vector{<:AbstractNode})
    activation = sum(weights(neuron) .* x, init=bias(neuron)) # wx + b
    out = tanh(activation)
    return out
end




struct Layer{N}
    neurons::Vector{Neuron}

    function Layer(number_inputs::Integer, number_outputs::Integer)
        return new{number_outputs}([Neuron(number_inputs) for _ in 1:number_outputs])
    end
end

neurons(layer::Layer) = getfield(layer, :neurons)
parameters(layer::Layer) = [p for neuron in neurons(layer) for p in parameters(neuron)]


function forward!(layer::Layer, x::Vector{<:AbstractNode}) #NOTE: do size checking for x!
    outs = [forward!(neuron, x) for neuron in neurons(layer)]
    # out = length(outs) == 1 ? outs[1] : outs #this is done so the output layer
    return outs
end



struct MLP{I, O} # Multi-Layer Perceptron
    layers::Vector{Layer}

    function MLP(number_inputs::Integer, number_outputs_list::Vector{<:Integer})
        sizes = vcat([number_inputs], number_outputs_list)
        return new{number_inputs, last(sizes)}([Layer(sizes[l], sizes[l + 1]) for l in 1:length(number_outputs_list)])
    end
end

layers(mlp::MLP) = getfield(mlp, :layers)
parameters(mlp::MLP) = [p for layer in layers(mlp) for p in parameters(layer)]

function forward!(mlp::MLP, x::Vector{<:AbstractNode})
    for layer in layers(mlp)
        x = forward!(layer, x)
    end
    return x #returns the output layer's out values
end
# function forward!(mlp::MLP, x::Vector{<:Real}) #might want to get rid of this
#     x = Node.(x)
#     for layer in layers(mlp)
#         x = forward!(layer, x)
#     end
#     return x #returns the output layer's out values
# end



"""
    Input(x::Vector{<:Real})

Special constructor for defining an input vector of Nodes
"""
# struct Input{N}
#     vals::
# end

const Input = Vector{LeafNode}
const Outputs = Vector{LeafNode}

# function Input(x::Vector{<:Real})
#     return Node.(x)
# end

# function Output(y::Vecot{<:Real})
#  return Float64.(y)
# end

struct TrainingSet{I, O} #(corresponds to Input, Output)
    xs::Vector{Input} #NOTE: these might always be LeafNodes
    ys::Outputs

    function TrainingSet(xs::Vector{<:Vector{<:Real}}, ys::Vector{<:Real})
        I = length(xs[1])
        @assert length(ys) == length(xs) "number of input samples must equal number of output samples"
        @assert all(v->length(v)==I, xs) "all input vectors must be the same length"
        # O = length(ys) #right now, O is always 1
        return new{I, 1}(broadcast(x->Node.(x), xs), Node.(ys))
    end
end

inputs(training_set::TrainingSet) = getfield(training_set, :xs)
outputs(training_set::TrainingSet) = getfield(training_set, :ys)


function train!(mlp::MLP{I, O}, training_set::TrainingSet{I, O}; steps::Integer=20, step_size::Float64=0.05) where {I, O}
    cost = 0.0 #initialize cost
    for step in 1:steps

        #forward pass
        ypred = [forward!(mlp, x) for x in inputs(training_set)]
        cost = sum([(yout[1] - ygt)^2 for (ygt, yout) in zip(outputs(training_set), ypred)]) #actually cost function?


        #reset all the gradients before doing the backward pass again
        for p in parameters(mlp)
            reset_gradient!(p)
        end
        
        #backward pass
        backprop!(cost)

        #update parameter values
        for p in parameters(mlp)
            accumulate_value!(p, step_size * -gradient(p))
        end

        println("$step: ", cost)
    end
end

function predict(mlp::MLP{I, O}, xs::Vector{<:Vector{<:Real}}) where {I, O}
    return [forward!(mlp, x) for x in broadcast(x->Node.(x), xs)]
end