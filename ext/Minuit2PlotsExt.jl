module Minuit2PlotsExt
using Minuit2
using FHist

isdefined(Base, :get_extension) ? (using Plots) : (using ..Plots)


"""
    draw_contour(m::Minuit,  x, y; bound=4, subtract_min=false, kwargs...)

Draw 2D minimal region (requires Plots module).

"""
function Minuit2.draw_contour(m::Minuit, x, y; bound=4, subtract_min=false, kwargs...)
    ix, xname = Minuit2.keypair(m, x)
    iy, yname = Minuit2.keypair(m, y)

    xv, yv, zv = Minuit2.contour(m, ix, iy, bound=bound, subtract_min=subtract_min)
    vx, vy = m.values[ix], m.values[iy]
    Plots.heatmap(xv, yv, zv', title="Contour $(m.funcname) for $(xname) vs $(yname)", xlabel=xname, ylabel=yname, kwargs...)
    Plots.scatter!([vx], [vy], label="Min", color=:white)
end

"""
    draw_mncontour(m::Minuit,  x, y; cl = 1:4, kwargs...)

Draw 2D Minos confidence region (requires Plots module).
"""
function Minuit2.draw_mncontour(m::Minuit, x, y; cl = 1:4, size=50, kwargs...)
    ix, xname = Minuit2.keypair(m, x)
    iy, yname = Minuit2.keypair(m, y)

    title = "MnContour $(m.funcname) for $(xname) vs $(yname)"
    plt = nothing

    for i in eachindex(cl)
        points = mncontour(m, ix, iy, cl=i, size=size)
        if i == 1
            plt = plot(points, title=title, label="$(cl[i]) σ", 
                 xlabel=xname, ylabel=yname, kwargs...)
        else
            plot!(points, label="$(cl[i]) σ", kwargs...)
        end
    end
    return plt
end

"""
    draw_profile(m::Minuit, var; band=true, text=true, kwargs...)

Draw 1D cost function profile over a range (requires matplotlib).

## Arguments
- `band::Bool=true` : If true, show a band to indicate the Hesse error interval.
- `text::Bool=true` : If true, show text a title with the function value and the Hesse error.
"""
function Minuit2.draw_profile(m::Minuit, var; band=true, text=true, kwargs...)
    ix, xname = Minuit2.keypair(m, var)
    x, y = Minuit2.profile(m, ix; subtract_min=true, kwargs...)
    v = m.values[ix]
    e = m.errors[ix]
    title = text ? "$xname= $(round(v, digits=3)) - $(round(e, digits=3)) + $(round(e, digits=3))" : nothing
    plt = plot(x, y; title=title, label=m.funcname, xlabel=xname, ylabel="FCN")
    vline!(plt, [v]; label=nothing, color=:black)
    band && vspan!(plt, [v - e, v + e]; label=nothing, color=:black, alpha=0.2)
    return plt
end

"""
    draw_mnprofile(m::Minuit, var; band=true, text=true, kwargs...)

Draw 1D Minos profile over a range (requires matplotlib).
"""
function Minuit2.draw_mnprofile(m::Minuit, var; band=true, text=true, kwargs...)
    ix, xname = Minuit2.keypair(m, var)
    x, y, _ = mnprofile(m, ix; subtract_min=true, kwargs...)
    v = m.values[ix]
    e = m.errors[ix]
    title = text ? "$xname= $(round(v, digits=3)) - $(round(e, digits=3)) + $(round(e, digits=3))" : nothing
    plt = plot(x, y; title=title, label=m.funcname, xlabel=xname, ylabel="FCN")
    vline!(plt, [v]; label=nothing, color=:black)
    band && vspan!(plt, [v - e, v + e]; label=nothing, color=:black, alpha=0.2)
    return plt
end

function Minuit2.visualize(m::Minuit; nbins=50, kwargs...)
    isnothing(m.cost) && throw(ArgumentError("Minuit object does not have a cost function"))
    m.cost.ndim > 1 && throw(ArgumentError("Cost function dimension > 1 not supported"))
    visualize(m.cost, m.is_valid, collect(m.values), nbins=nbins, kwargs...)
end
function visualize(cost::LeastSquares, is_valid, pars; nbins=50, kwargs...)
    x = cost.x
    y = cost.y
    yerr = cost.yerror
    plt = plot(x, y, yerr=yerr, seriestype=:scatter, kwargs...)
    if is_valid
        yt = cost.model.(x, pars...)
        plt = plot!(plt, x, yt; label="Fit")
    end
    return plt
end
function visualize(cost::UnbinnedCostFunction, is_valid, pars; nbins=50, kwargs...)
    h = Hist1D(cost.data, nbins=nbins)
    x = bincenters(h)
    y = bincounts(h)
    dy = sqrt.(y)
    plt = plot(x, y, yerr=dy, seriestype=:scatter, label="Data")
    if is_valid
        if cost isa UnbinnedNLL
            scale = prod(Base.size(cost.data))*(x[2]-x[1])
            plot!(plt, x -> cost.model(x, pars...)*scale; label="Fit")
        else
            scale = (x[2]-x[1])
            plot!(plt, x -> cost.model(x, pars...)[2]*scale; label="Fit")
        end                
    end
    return plt
end
function visualize(cost::BinnedCostFunction, is_valid, pars; nbins=50, kwargs...)
    x = [0.5*(cost.binedges[i] + cost.binedges[i+1]) for i in 1:length(cost.binedges)-1]
    dx = (x[2]-x[1])/2
    y = cost.bincounts
    dy = sqrt.(y)
    plt = plot(x, y, yerr=dy, seriestype=:scatter, label="Data")
    if is_valid
        if cost isa BinnedNLL
            scale = sum(cost.bincounts)
        else
            scale = 1
        end 
        if cost.use_pdf == :approximate
            f = x -> cost.model(x, pars...)*scale*2dx
        else
            f = x -> (cost.model(x+dx, pars...)-cost.model(x-dx, pars...))*scale
        end
        plot!(plt, f; label="Fit")
    end
    return plt
end
function visualize(cost::Minuit2.CostSum, is_valid, pars; nbins=50, kwargs...)
    plots = []
    for (i, c) in enumerate(cost.costs)
        plt = visualize(c, is_valid, pars[cost.argsmapping[i]]; nbins=nbins, kwargs...)
        push!(plots, plt)
    end
    n = plots |> length |> sqrt |> ceil |> Int 
    plt = plot(plots...; layout=(n, n))
    return plt
end
end