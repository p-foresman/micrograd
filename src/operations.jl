abstract type Operation end
struct Null <: Operation end #for leaf nodes
struct Add <: Operation end
struct Mul <: Operation end
struct Tanh <: Operation end
struct Exp <: Operation end
struct Pow <: Operation end
struct Div <: Operation end
