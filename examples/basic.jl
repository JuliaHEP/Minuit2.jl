using Revise
using Minuit2
using Plots


# let's make a line model
line(x, α, β) = α + x * β
grad_line(x, α, β) = [1.0, x]

a_true = 1.0
b_true = 2.0
sigma_y = 0.1

const npoints = 10
# let's make some data
x = range(0, 1., npoints) |> collect
y = line.(x, a_true, b_true) + 0.1 * randn(length(x))


plot(x, y, yerr=sigma_y, seriestype=:scatter)

cost = LeastSquares(x, y, fill(sigma_y, length(x)), line, model_grad=grad_line)
value(cost, (1.0, 2.0))

m = Minuit(cost, [0.0, 0.0], grad=true)
migrad!(m)
show(m)
println("nfcn = $(m.fcn.nfcn) ngrad = $(m.fcn.ngrad)")



