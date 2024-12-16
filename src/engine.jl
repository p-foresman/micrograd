abstract type Operation end
struct Null <: Operation end #for leaf nodes
struct Add <: Operation end
struct Mul <: Operation end
struct Tanh <: Operation end

mutable struct Value{O}
    data::Float64
    gradient::Float64
    prev::Tuple{Vararg{Value}} #make this set?
    label::String #not using yet

    function Value(data::Number, children::Tuple{Vararg{Value}}=(), operation::Type{<:Operation}=Null, label::String="")
        return new{operation}(data, 0.0, children, label)
    end
end

data(value::Value) = getfield(value, :data)
gradient(value::Value) = getfield(value, :gradient)
gradient!(value::Value, grad::Number) = setfield!(value, :gradient, Float64(grad))
prev(value::Value) = getfield(value, :prev)
function prev(value::Value, index::Integer)
    try
        return getindex(prev(value), index)
    catch err
        if err isa BoundsError
            return nothing
        end
    end
end
label(value::Value) = getfield(value, :label)
backprop(value::Value) = getfield(value, :backprop)() #call the function

Base.show(io::IO, value::Value{O}) where {O} = print(io, "Value{$O}(data=$(data(value)), gradient=$(gradient(value)))")

import Base: +, *
+(value1::Value, value2::Value) = Value{Add}(data(value1) + data(value2), (value1, value2), Add)
*(value1::Value, value2::Value) = Value{Mul}(data(value1) * data(value2), (value1, value2), Mul)
function tanh(value::Value)
    x = data(value)
    t = (exp(2*x) - 1)/(exp(2*x) + 1)
    return Value{Tanh}(t, (value,), Tanh)
end

backprop(::Value{Null}) = nothing

function backprop(value::Value{Add})
    gradient!(prev(value, 1), 1.0 * gradient(value))
    gradient!(prev(value, 2), 1.0 * gradient(value))
    # for child in prev(value)
    #     gradient!(child, 1.0 * gradient(value))
    # end
end

function backprop(value::Value{Mul})
    gradient!(prev(value, 1), data(prev(value, 2)) * gradient(value))
    gradient!(prev(value, 2), data(prev(value, 1)) * gradient(value))
end

function backprop(value::Value{Tanh})
    gradient!(prev(value, 1), (1 - data(value)^2) * gradient(value))
end