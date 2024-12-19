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
Base.:+(node1::Node, node2::Node) = Node{Add}(data(node1) + data(node2), (node1, node2))
Base.:*(node1::Node, node2::Node) = Node{Mul}(data(node1) * data(node2), (node1, node2))
function Base.tanh(node::Node)
    x = data(node)
    t = (exp(2*x) - 1)/(exp(2*x) + 1)
    return Node{Tanh}(t, (node,))
end
Base.exp(node::Node) = Node{Exp}(exp(data(node)), (node,))
Base.:^(node::Node, power::Real) = Node{Pow}(data(node)^power, (node,))
Base.:/(node1::Node, node2::Node) = Node{Div}(node1 * (node2^-1), (node1, node2))



################# backpropogation ####################

function backprop(node::Node)
    backprop_func(node)
    for n in prev(node)
        backprop(n)
    end
end
backprop(::Node{Null}) = nothing

function backprop_func(node::Node{Add})
    accumulate_gradient!(prev(node, 1), 1.0 * gradient(node))
    accumulate_gradient!(prev(node, 2), 1.0 * gradient(node))
end

function backprop_func(node::Node{Mul})
    accumulate_gradient!(prev(node, 1), data(prev(node, 2)) * gradient(node))
    accumulate_gradient!(prev(node, 2), data(prev(node, 1)) * gradient(node))
end

function backprop_func(node::Node{Tanh})
    accumulate_gradient!(prev(node, 1), (1 - data(node)^2) * gradient(node))
end

function backprop_func(node::Node{Exp})
    accumulate_gradient!(prev(node, 1), data(node) * gradient(node))
end

function backprop_func(node::Node{Pow}) #Note: shit, no access to the power used to create this node!
    accumulate_gradient!(prev(node, 1),  * gradient(node))
end