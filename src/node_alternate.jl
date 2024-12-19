"""
    Node{O} where {O<:Operation}

Node type used as a node in a function tree. O is the Operation type used to construct the Node instance
"""
mutable struct Node{O} #<: Real
    data::Float64
    gradient::Float64
    prev::Tuple{Vararg{Node}} #make this set?
    label::String #not using yet

    function Node{O}(data::Real, children::Tuple{Vararg{Node}}=(); label::String="") where {O<:Operation} # operation::Type{<:Operation}=Null
        return new{O}(data, 0.0, children, label)
    end
end

function Node(data::Real; label::String="") #constructor for the Null case (leaf node)
    return Node{Null}(data; label=label)
end


# Node(node::Node) = node #multiple dispatch to ensure a given datum (Node or Real) is a Node

######### Node methods #############
data(node::Node) = getfield(node, :data)
gradient(node::Node) = getfield(node, :gradient)
gradient!(node::Node, grad::Real) = setfield!(node, :gradient, Float64(grad))
accumulate_gradient!(node::Node, grad::Real) = gradient!(node, gradient(node) + grad)
prev(node::Node) = getfield(node, :prev)
function prev(node::Node, index::Integer)
    try
        return getindex(prev(node), index)
    catch err
        if err isa BoundsError
            return nothing
        end
    end
end
label(node::Node) = getfield(node, :label)
label!(node::Node, label::String) = setfield!(node, :label, label)

Base.show(io::IO, node::Node{O}) where {O} = print(io, "Node{$O}(data=$(data(node)), gradient=$(gradient(node)))")