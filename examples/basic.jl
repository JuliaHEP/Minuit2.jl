using Revise
using Minuit2
#using Plots

#rosenbrock(x, y; a=1, b=100) = (a - x)^2 + b*(y - x^2)^2
rosenbrock(v; a=1, b=100) = (a - v[1])^2 + b*(v[2] - v[1]^2)^2

#=
# Define the range for x and y
x = -2:0.01:2
y = -1:0.01:3

# Create a grid of (x, y) points
z = [rosenbrock(xi, yi) for xi in x, yi in y];

# Plot the function with contour lines and log scale heatmap using Plots.jl
heatmap(x, y, log10.(z'), xlabel="x", ylabel="y", title="Rosenbrock Function (Log Scale)", color=:viridis)
scatter!([1.0], [1.0], label="Minimum", color=:red, markersize=3)
=#

#ROOT!Minuit2!MnPrint!SetGlobalLevel(2)

m = Minuit(rosenbrock, [0.0, 0.0])
migrad!(m)
values(m)


#res = minuit(rosenbrock, [0.0, 0.0]; names=["a", "b"])

#=
minuit2.method("fit_Migrad", [](JuliaFcn& fcn, jlcxx::ArrayRef<double> pars, jlcxx::ArrayRef<double> errs) {
    std::vector<double> parameters;
    for (auto p : pars) parameters.push_back(p);
    std::vector<double> errors;
    for (auto e : errs) errors.push_back(e);

    MnUserParameters upar(parameters, errors);
    MnMigrad migrad(fcn, upar);
    FunctionMinimum min = migrad();

    const double* data = min.Parameters().Vec().Data();
    for (size_t idx=0; idx<pars.size(); ++idx) {
        pars[idx] = data[idx]; 
    }
});
=#


function get_argument_names(rosenbrock)



using Optim
res = optimize(rosenbrock, [0.0, 0.0])
res.minimizer


Optim.minimizer(res)


