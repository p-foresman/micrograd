############ operation extension methods ############

# function Base.:+(node1::Union{Node, Number}, node2::Union{Node, Number})
#     node1 = Node(node1) #ensure these are Nodes, not Numbers
#     node2 = Node(node2)
#     return Node(data(node1) + data(node2), (node1, node2), Add)
# end
# function Base.:*(node1::Union{Node, Number}, node2::Union{Node, Number})
#     node1 = Node(node1) #ensure these are Nodes, not Numbers
#     node2 = Node(node2)
#     return Node(data(node1) * data(node2), (node1, node2), Mul)
# end
# function Base.tanh(node::Union{Node, Number})
#     node = Node(node)
#     x = data(node)
#     t = (exp(2*x) - 1)/(exp(2*x) + 1)
#     return Node(t, (node,), Tanh)
# end
Base.:+(node1::AbstractNode, node2::AbstractNode) = AddNode(node1, node2)
Base.:*(node1::AbstractNode, node2::AbstractNode) = MulNode(node1, node2)
Base.tanh(node::AbstractNode) = TanhNode(node)
Base.exp(node::AbstractNode) = ExpNode(node)
Base.:^(node::AbstractNode, power::Real) = PowNode(node, power)
Base.:/(node1::AbstractNode, node2::AbstractNode) = MulNode(node1, PowNode(node2, -1)) #create division from other node types (node1 * (node2^-1))
Base.:-(node1::AbstractNode, node2::AbstractNode) = AddNode(node1, MulNode(node2, Node(-1))) #create subtraction from other node types (node1 + (-1*node2))



################# backpropogation ####################

# function backprop(node::AbstractNode)
#     backprop_func(node)
#     for n in prev(node)
#         backprop(n)
#     end
# end

# backprop(::LeafNode, ::Vector) = nothing

# function backprop(node::AddNode, visited::Vector{<:AbstractNode})
#     if !(node in visited)
#         push!(visited, node)
#         accumulate_gradient!(prev(node)[1], 1.0 * gradient(node))
#         accumulate_gradient!(prev(node)[2], 1.0 * gradient(node))
#         for n in prev(node) backprop(n, visited) end #could get rid of this of all prevs are tuples
#     end
# end

# function backprop(node::MulNode, visited::Vector{<:AbstractNode})
#     if !(node in visited)
#         push!(visited, node)
#         accumulate_gradient!(prev(node)[1], value(prev(node)[2]) * gradient(node))
#         accumulate_gradient!(prev(node)[2], value(prev(node)[1]) * gradient(node))
#         for n in prev(node) backprop(n, visited) end
#     end
# end

# function backprop(node::TanhNode, visited::Vector{<:AbstractNode})
#     if !(node in visited)
#         push!(visited, node)
#         accumulate_gradient!(prev(node), (1 - value(node)^2) * gradient(node))
#         backprop(prev(node), visited)
#     end
# end

# function backprop(node::ExpNode, visited::Vector{<:AbstractNode})
#     if !(node in visited)
#         push!(visited, node)
#         accumulate_gradient!(prev(node), value(node) * gradient(node))
#         backprop(prev(node), visited)
#     end
# end

# function backprop(node::PowNode, visited::Vector{<:AbstractNode})
#     if !(node in visited)
#         push!(visited, node)
#         accumulate_gradient!(prev(node), power(node) * (value(prev(node)) ^ (power(node) - 1)) * gradient(node))
#         backprop(prev(node), visited)
#     end
# end

# function backprop(node::AbstractNode)
#     visited = []
#     backprop(node, visited)
# end






# _backprop(::LeafNode, ::Vector) = nothing

# function _backprop(node::AddNode, visited::Vector{<:AbstractNode})
#     println(visited)
#     if !(prev(node)[1] in visited)
#         push!(visited, prev(node)[1])
#         accumulate_gradient!(prev(node)[1], 1.0 * gradient(node))
#         # _backprop(prev(node)[1], visited) #could get rid of this of all prevs are tuples
#     end
#     if !(prev(node)[2] in visited)
#         push!(visited, prev(node)[2])
#         accumulate_gradient!(prev(node)[2], 1.0 * gradient(node))
#         # _backprop(prev(node)[2], visited) #could get rid of this of all prevs are tuples
#     end
#     if !(prev(node)[1] in visited) _backprop(prev(node)[1], visited) end
#     if !(prev(node)[2] in visited) _backprop(prev(node)[2], visited) end
# end

# function _backprop(node::MulNode, visited::Vector{<:AbstractNode})
#     println(visited)
#     if !(prev(node)[1] in visited)
#         push!(visited, prev(node)[1])
#         accumulate_gradient!(prev(node)[1], value(prev(node)[2]) * gradient(node))
#         # _backprop(prev(node)[1], visited) #could get rid of this of all prevs are tuples
#     end
#     if !(prev(node)[2] in visited)
#         push!(visited, prev(node)[2])
#         accumulate_gradient!(prev(node)[2], value(prev(node)[1]) * gradient(node))
#         # _backprop(prev(node)[2], visited) #could get rid of this of all prevs are tuples
#     end
#     if !(prev(node)[1] in visited) _backprop(prev(node)[1], visited) end
#     if !(prev(node)[2] in visited) _backprop(prev(node)[2], visited) end
# end

# function _backprop(node::TanhNode, visited::Vector{<:AbstractNode})
#     println(visited)
#     if !(prev(node) in visited)
#         push!(visited, prev(node))
#         accumulate_gradient!(prev(node), (1 - value(node)^2) * gradient(node))
#         _backprop(prev(node), visited)
#     end
# end

# function _backprop(node::ExpNode, visited::Vector{<:AbstractNode})
#     println(visited)
#     if !(prev(node) in visited)
#         push!(visited, prev(node))
#         accumulate_gradient!(prev(node), value(node) * gradient(node))
#         _backprop(prev(node), visited)
#     end
# end

# function _backprop(node::PowNode, visited::Vector{<:AbstractNode})
#     println(visited)
#     if !(prev(node) in visited)
#         push!(visited, prev(node))
#         accumulate_gradient!(prev(node), power(node) * (value(prev(node)) ^ (power(node) - 1)) * gradient(node))
#         _backprop(prev(node), visited)
#     end
# end

function backprop(node::AbstractNode)
    visited = Set()

    function dfs(v)
        if !(v in visited)
            push!(visited, v)
            println(v)
            if v isa LeafNode
                nothing
            elseif v isa AddNode
                accumulate_gradient!(prev(v)[1], 1.0 * gradient(v))
                accumulate_gradient!(prev(v)[2], 1.0 * gradient(v))
                for child in prev(v)
                    dfs(child)
                end
            elseif v isa MulNode
                accumulate_gradient!(prev(v)[1], value(prev(v)[2]) * gradient(v))
                accumulate_gradient!(prev(v)[2], value(prev(v)[1]) * gradient(v))
                for child in prev(v)
                    dfs(child)
                end
            elseif v isa TanhNode
                accumulate_gradient!(prev(v), (1 - (value(v)^2)) * gradient(v))
                dfs(prev(v))
            elseif v isa ExpNode
                accumulate_gradient!(prev(v), value(v) * gradient(v))
                dfs(prev(v))
            elseif v isa PowNode
                accumulate_gradient!(prev(v), power(v) * (value(prev(v)) ^ (power(v) - 1)) * gradient(v))
                dfs(prev(v))
            end
        end
    end

    gradient!(node, 1)
    dfs(node)
end