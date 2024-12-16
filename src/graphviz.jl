using GraphViz

function trace(root)
    nodes, edges = Set(), Set()
    function build(v)
        if !(v in nodes)
            push!(nodes, v)
            for child in prev(v)
                push!(edges, (child, v))
                build(child)
            end
        end
    end
    build(root)
    return nodes, edges
end

function draw_dot(root, format="svg", rankdir="LR")
    @assert rankdir in ["LR", "TB"]
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr=Dict("rankdir"=>rankdir)) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes
        dot.node(name=str(id(n)), label = "{ data $(data(n)) | grad $(grad(n)) }", shape="record")
        if operation(n)
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
        end
    end

    for (n1, n2) in edges
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    end
    
    return dot
end