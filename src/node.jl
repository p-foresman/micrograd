abstract type AbstractNode end

mutable struct _Data #Note: should be private
    value::Float64
    gradient::Float64
    label::String

    function _Data(value::Real, label::String)
        return new(Float64(value), 0.0, label)
    end
end

struct LeafNode <: AbstractNode
    data::_Data

    function LeafNode(value::Real; label::String="")
        return new(_Data(value, label))
    end
end
Node(value::Real; label::String="") = LeafNode(value; label=label) #general 'AbstractNode' constructor should be LeafNode

struct AddNode <: AbstractNode
    data::_Data
    prev::Tuple{AbstractNode, AbstractNode} #make this set?

    function AddNode(node1::AbstractNode, node2::AbstractNode; label::String="")
        return new(_Data(value(node1) + value(node2), label), (node1, node2))
    end
end
struct MulNode <: AbstractNode
    data::_Data
    prev::Tuple{AbstractNode, AbstractNode} #make this set?

    function MulNode(node1::AbstractNode, node2::AbstractNode; label::String="")
        return new(_Data(value(node1) * value(node2), label), (node1, node2))
    end
end
struct TanhNode <: AbstractNode
    data::_Data
    prev::AbstractNode

    function TanhNode(node::AbstractNode; label::String="")
        x = value(node)
        t = (exp(2*x) - 1)/(exp(2*x) + 1)
        return new(_Data(t, label), node)
    end
end
struct ExpNode <: AbstractNode
    data::_Data
    prev::AbstractNode #make this set?

    function ExpNode(node::AbstractNode; label::String="")
        return new(_Data(exp(value(node)), label), node)
    end
end
struct PowNode <: AbstractNode
    data::_Data
    prev::AbstractNode #make this set?
    power::Float64

    function PowNode(node::AbstractNode, power::Real; label::String="")
        return new(_Data(value(node)^power, label), node, Float64(power))
    end
end


######### AbstractNode methods #############
_data(node) = getfield(node, :data)
value(node::AbstractNode) = getfield(_data(node), :value)
value!(node::AbstractNode, val::Real) = setfield!(_data(node), :value, Float64(val))
gradient(node::AbstractNode) = getfield(_data(node), :gradient)
gradient!(node::AbstractNode, grad::Real) = setfield!(_data(node), :gradient, Float64(grad))
accumulate_gradient!(node::AbstractNode, grad::Real) = gradient!(node, gradient(node) + grad)
prev(node::AbstractNode) = getfield(node, :prev) #NOTE: will likely cause type instability
# function prev(node::AbstractNode, index::Integer)
#     try
#         return getindex(prev(node), index)
#     catch err
#         if err isa BoundsError
#             return nothing
#         end
#     end
# end
label(node::AbstractNode) = getfield(_data(node), :label)
label!(node::AbstractNode, label::String) = setfield!(_data(node), :label, label)

power(node::PowNode) = getfield(node, :power)

Base.show(io::IO, node::AbstractNode) = print(io, "$(typeof(node))(value=$(value(node)), gradient=$(gradient(node)))") #NOTE: fix this?