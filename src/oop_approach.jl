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
    prev::Tuple{}
    backprop::Function

    function LeafNode(value::Real; label::String="")
        return new(_Data(value, label), (), ()->nothing)
    end
end
Node(value::Real; label::String="") = LeafNode(value; label=label) #general 'AbstractNode' constructor should be LeafNode

struct AddNode <: AbstractNode
    data::_Data
    prev::Tuple{AbstractNode, AbstractNode} #make this set?
    backprop::Function

    function AddNode(node1::AbstractNode, node2::AbstractNode; label::String="")
        out_data = _Data(value(node1) + value(node2), label)
        function bp()
            accumulate_gradient!(node1, 1.0 * gradient(out_data))
            accumulate_gradient!(node2, 1.0 * gradient(out_data))
            return nothing
        end
        return new(out_data, (node1, node2), bp)
    end
end

mutable struct MulNode <: AbstractNode
    data::_Data
    prev::Tuple{AbstractNode, AbstractNode} #make this set?
    backprop::Function

    function MulNode(node1::AbstractNode, node2::AbstractNode; label::String="")
        out_data = _Data(value(node1) * value(node2), label)
        function bp()
            accumulate_gradient!(node1, value(node2) * gradient(out_data))
            accumulate_gradient!(node2, value(node1) * gradient(out_data))
            return nothing
        end
        return new(out_data, (node1, node2), bp)
    end
end
mutable struct TanhNode <: AbstractNode
    data::_Data
    prev::Tuple{AbstractNode}
    backprop::Function

    function TanhNode(node::AbstractNode; label::String="")
        x = value(node)
        t = (exp(2*x) - 1)/(exp(2*x) + 1)
        out_data = _Data(t, label)
        function bp()
            accumulate_gradient!(node, (1 - (t^2)) * gradient(out_data))
            return nothing
        end
        return new(out_data, (node,), bp)
    end
end
mutable struct ExpNode <: AbstractNode
    data::_Data
    prev::Tuple{AbstractNode}
    backprop::Function

    function ExpNode(node::AbstractNode; label::String="")
        out_data = _Data(exp(value(node)), label)
        function bp()
            accumulate_gradient!(node, value(out_data) * gradient(out_data))
            return nothing
        end
        return new(out_data, (node,), bp)
    end
end
mutable struct PowNode <: AbstractNode
    data::_Data
    prev::Tuple{AbstractNode}
    power::Float64
    backprop::Function

    function PowNode(node::AbstractNode, power::Real; label::String="")
        out_data = _Data(value(node)^power, label)
        function bp()
            accumulate_gradient!(node, power * (value(node) ^ (power - 1)) * gradient(out_data))
            return nothing
        end
        return new(out_data, (node,), power, bp)
    end
end


######### AbstractNode methods #############
_data(node) = getfield(node, :data)
value(node::AbstractNode) = getfield(_data(node), :value)
value!(node::AbstractNode, val::Real) = setfield!(_data(node), :value, Float64(val))
gradient(node::AbstractNode) = getfield(_data(node), :gradient)
gradient!(node::AbstractNode, grad::Real) = setfield!(_data(node), :gradient, Float64(grad))
accumulate_gradient!(node::AbstractNode, grad::Real) = gradient!(node, gradient(node) + grad)

value(data::_Data) = getfield(data, :value)
gradient(data::_Data) = getfield(data, :gradient)
gradient!(data::_Data, grad::Real) = setfield!(data, :gradient, Float64(grad))
accumulate_gradient!(data::_Data, grad::Real) = gradient!(data, gradient(node) + grad)

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

do_backprop(node::AbstractNode) = getfield(node, :backprop)()

power(node::PowNode) = getfield(node, :power)

Base.show(io::IO, node::AbstractNode) = print(io, "$(typeof(node))(value=$(value(node)), gradient=$(gradient(node)), label=$(label(node)))") #NOTE: fix this?











Base.:+(node1::AbstractNode, node2::AbstractNode) = AddNode(node1, node2)
Base.:*(node1::AbstractNode, node2::AbstractNode) = MulNode(node1, node2)
Base.tanh(node::AbstractNode) = TanhNode(node)
Base.exp(node::AbstractNode) = ExpNode(node)
Base.:^(node::AbstractNode, power::Real) = PowNode(node, power)
Base.:/(node1::AbstractNode, node2::AbstractNode) = MulNode(node1, PowNode(node2, -1)) #create division from other node types (node1 * (node2^-1))
Base.:-(node1::AbstractNode, node2::AbstractNode) = AddNode(node1, MulNode(node2, Node(-1))) #create subtraction from other node types (node1 + (-1*node2))



function backprop(node::AbstractNode)
    topo = []
    visited = Set()
    function build_topo!(v)
        if !(v in visited)
            push!(visited, v)
            for child in prev(v)
                build_topo!(child)
            end
            push!(topo, v)
        end
    end
    build_topo!(node)

    gradient!(node, 1)
    for node in reverse(topo)
        # println(node)
        do_backprop(node)
    end
end


function backprop_kahns(node::AbstractNode)
    q = Queue{AbstractNode}()
    gradient!(node, 1)
    enqueue!(q, node)

    while !isempty(q)
        curr = dequeue!(q)
        depends = Vector{AbstractNode}() #this step is sketchy, as im building a dependency list every iteration
        for n in q
            for p in prev(n)
                push!(depends, p) 
            end
        end
        if !(curr in depends) #if the current node has nothing depending on it, do backprop. The node will not be requeued.
            do_backprop(curr)
            # println(curr)
            for child in prev(curr)
                if !(child in q)
                    enqueue!(q, child)
                end
            end
        else
            enqueue!(q, curr) # if it had something depending on it, requeue it. This ensures reverse topological order
        end
    end
end