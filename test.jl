using Micrograd

# x1 = Node(2, label="x1")
# x2 = Node(0, label="x2")
# w1 = Node(-3, label="w1")
# w2 = Node(1, label="w2")
# b = Node(6.8813735670195432, label="b")
# x1w1 = x1*w1; label!(x1w1, "x1w1")
# x2w2 = x2*w2; label!(x2w2, "x2w2")
# x1w1x2x2 = x1w1 + x2w2; label!(x1w1x2x2, "x1w1 + x2w2")
# n = x1w1x2x2 + b; label!(n, "n")
# o = tanh(n)

# # gradient!(o, 1)
# backprop(o)




x1 = Node(2, label="x1")
x2 = Node(0, label="x2")
w1 = Node(-3, label="w1")
w2 = Node(1, label="w2")
b = Node(6.8813735670195432, label="b")
x1w1 = x1*w1; label!(x1w1, "x1w1")
x2w2 = x2*w2; label!(x2w2, "x2w2")
x1w1x2x2 = x1w1 + x2w2; label!(x1w1x2x2, "x1w1 + x2w2")
n = x1w1x2x2 + b; label!(n, "n")
o = ((exp(Node(2)*n)) - Node(1)) / ((exp(Node(2)*n)) + Node(1))


Micrograd.backprop_kahns(o)




# a1 = Node(2)
# o1 = tanh(a1)
# gradient!(o1, 1)
# backprop(o1)
# gradient(a1) #0.07065082485316443


# a2 = Node(2)
# e = exp(Node(2)*a2)
# o2 = ((exp(Node(2)*a2)) - Node(1)) / ((exp(Node(2)*a2)) + Node(1))
# gradient!(o2, 1)
# backprop(o2)
# gradient(a2) #0.07065082485316443


