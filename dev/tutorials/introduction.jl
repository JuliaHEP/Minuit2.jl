using Minuit2
using Polynomials
using Plots

# let's make a line model
line(x, a, b) = a + x * b

a_true = 1.0
b_true = 2.0

# let's make some data
x = range(0, 1., 10)

# precomputed random numbers from standard normal distribution
z = [-0.49783783, -0.33041722, -1.71800806,  1.60229399,
      1.36682387, -1.15424221, -0.91425267, -0.03395604,
      -1.27611719, -0.7004073 ]

sigma_y = 0.1
y = line.(x, a_true, b_true) + sigma_y * z

# Plot with error bars
plot(x, y, yerr=sigma_y, seriestype=:scatter)

LSQ(a, b) = sum((y - line.(x, a, b)) .^ 2 ./ sigma_y .^ 2)

m = Minuit(LSQ; a=2, b=5, fix_a=true,
                error_a=0.1, error_b=0.1,
                errordef=1)

migrad!(m)

# get parameter values
a_fit, b_fit = m.values

# Plot with error bars
plot(x, y, yerr=sigma_y, seriestype=:scatter)
plot!(x, line.(x, a_fit, b_fit))

m = Minuit(LSQ; a=5, b=5, error_a=0.1, error_b=0.1,
                limit_a=(0, Inf), limit_b=(0, 10.), errordef=1)
migrad!(m)

# get parameter values
a_fit, b_fit = m.values

# plot with error bars
plot(x, y, yerr=sigma_y, seriestype=:scatter)
plot!(x, line.(x, a_fit, b_fit))

function LSQ_v(par)       # par is a vector of parameters
    pol = Polynomial(par) # for len(par) == 2 this is a line
    sum((y - pol.(x)) .^ 2 ./ sigma_y .^ 2)
end
#  This is the order of coefficients in the polynomial (reverse order in np.polyval)
Polynomial([1,2,3,4])

m = Minuit(LSQ_v, [5, 5], error=[0.1, 0.1], errordef=1)
migrad!(m)

m = Minuit(LSQ_v, [2, 1, 3, 5], error=0.1,
           names=("a", "b", "c", "d"), errordef=1)
migrad!(m)

# get parameter values
par_fit = m.values
pol = Polynomial(par_fit)

# Plot with error bars
plot(x, y, yerr=sigma_y, seriestype=:scatter)
plot!(x, line.(x, a_fit, b_fit), label="pol2")
plot!(x, pol.(x), label="pol4")

m.fval / (length(y) - length(par_fit))

hesse!(m)

matrix(m)

matrix(m, correlation=true)

minos!(m)

m.minos |> show

d = m.minos["d"]
d.lower, d.upper

v = m.values |> collect
ve = m.errors |> collect
vm = m.minos |> values
vmean = [(e.upper - e.lower) / 2 for e in vm]

npar = length(v)
indices = 1:npar

plot(indices .- 0.05, v, yerr=ve, seriestype=:scatter, label="Hesse errors")
plot!(indices .+ 0.05, v, yerr=vmean, seriestype=:scatter, label="Minos errors")

draw_contour(m, "d", "c", bound=4, subtract_min=false)

draw_mncontour(m, "d", "c", cl=1:4)

draw_profile(m, "d", bound=2)

draw_mnprofile(m, "d", size=20)
