############ operation extension methods ############

Base.:+(node1::AbstractNode, node2::AbstractNode) = AddNode(node1, node2)
Base.:*(node1::AbstractNode, node2::AbstractNode) = MulNode(node1, node2)
Base.tanh(node::AbstractNode) = TanhNode(node)
Base.exp(node::AbstractNode) = ExpNode(node)
Base.:^(node::AbstractNode, power::Real) = PowNode(node, power)
Base.:/(node1::AbstractNode, node2::AbstractNode) = MulNode(node1, PowNode(node2, -1)) #create division from other node types (node1 * (node2^-1))
Base.:-(node1::AbstractNode, node2::AbstractNode) = AddNode(node1, MulNode(node2, Node(-1))) #create subtraction from other node types (node1 + (-1*node2))
# relu(node::AbstractNode) = ReluNode(node)



################# backpropogation ####################

function backprop!(node::AbstractNode) #using a variation of Kahn's algorithm to traverse and backprop in a topological order
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
            _backprop!(curr)
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

_backprop!(::LeafNode) = nothing

function _backprop!(node::AddNode)
    accumulate_gradient!(prev(node)[1], 1.0 * gradient(node))
    accumulate_gradient!(prev(node)[2], 1.0 * gradient(node))
end

function _backprop!(node::MulNode)
    accumulate_gradient!(prev(node)[1], value(prev(node)[2]) * gradient(node))
    accumulate_gradient!(prev(node)[2], value(prev(node)[1]) * gradient(node))
end

function _backprop!(node::TanhNode)
    accumulate_gradient!(prev(node)[1], (1 - value(node)^2) * gradient(node))
end

function _backprop!(node::ExpNode)
    accumulate_gradient!(prev(node)[1], value(node) * gradient(node))
end

function _backprop!(node::PowNode)
    accumulate_gradient!(prev(node)[1], power(node) * (value(prev(node)[1]) ^ (power(node) - 1)) * gradient(node))
end

# function _backprop!(node::ReluNode)
#     accumulate_gradient!(prev(node)[1], (value(node) > 0) * gradient(node))
# end