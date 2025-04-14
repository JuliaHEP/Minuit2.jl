module Minuit2PlotsExt
using Minuit2
using Minuit2.RooFit
using FHist, RecipesBase, Plots

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
    visualize(m.cost, m.is_valid, collect(m.values); nbins=nbins, kwargs...)
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
    plt = plot(x, y ;yerr=dy, seriestype=:scatter, label="Data", kwargs...)
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
    x = cost.bincenters
    dx = (x[2]-x[1])/2
    y = cost.bincounts
    dy = sqrt.(y)
    plt = plot(x, y; yerr=dy, seriestype=:scatter, label="Data", kwargs...)
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

#----Visualize fit and model
function Minuit2.visualize(m::Minuit, model::AbstractPdf; nbins=-1, components=(), kwargs...)
    isnothing(m.cost) && throw(ArgumentError("Minuit object does not have a cost function"))
    m.cost.ndim > 1 && throw(ArgumentError("Cost function dimension > 1 not supported"))
    m.cost isa UnbinnedCostFunction && nbins == -1 && (nbins = 50)
    plt = visualize(m.cost, m.is_valid, collect(m.values), nbins=nbins; kwargs...)
    #---Components of the model
    for c in components
        comp, weight = model[c]
        a, b = comp.x.limits
        if m.cost isa UnbinnedNLL
            nbins == -1 && (nbins = 50)
            scale = weight * prod(Base.size(m.cost.data)) * (b-a)/nbins
        elseif m.cost isa ExtendedUnbinnedNLL
            nbins == -1 && (nbins = 50)
            scale = weight * (b-a)/nbins
        elseif m.cost isa BinnedNLL
            nbins != -1 && nbins != comp.x.nbins && @warn "Forced #bins to $(comp.x.nbins)"
            scale = weight * (b-a)/comp.x.nbins * sum(m.cost.bincounts)
        elseif m.cost isa ExtendedBinnedNLL
            nbins != -1 && nbins != comp.x.nbins && @warn "Forced #bins to $(comp.x.nbins)"
            scale = weight * (b-a)/comp.x.nbins
        end
        func = x -> comp.pdf(x, (p.value for p in comp.params)...) * scale
        plot!(plt, func; label="$(comp.name)", kwargs...)
    end
    return plt
end
#---Visualize PDF model
function Minuit2.visualize(model::AbstractPdf; kwargs...)
    isnothing(model.x) && throw(ArgumentError("Model does not have a variable"))
    plt = plot(x->model(x), model.x.limits...; label="$(model.name)", kwargs...)
    return plt
end

#---Plots.jl recipes-------------------------------------------------------------------------------
global plot_attributes = Dict()    # a hacky way to pass attributes between recipes
info(model) = join(["$(p.name) = $(round(p.value, sigdigits=3)) ± $(round(p.error, sigdigits=3))" for p in model.params], '\n')

@recipe function f(pdf::AbstractPdf; components=())
    seriestype --> :path
    scale = get(plotattributes, :integral, get(plot_attributes, :integral, 1.))
    @series begin
        pdf_int = pdf isa AddPdf && pdf.extendable ? sum(f.value for f in pdf.fractions) : 1
        linestyle := :solid
        components != () && (label := string(pdf.name))
        (x -> pdf.pdf(x, (p.value for p in pdf.params)...)*scale/pdf_int, pdf.x.limits...)
    end
    # Plot the components of the PDF
    for c in components
        comp, weight = pdf[c]
        a, b = comp.x.limits
        func = x -> comp.pdf(x, (p.value for p in comp.params)...)*scale*weight
        @series begin
            label := string(comp.name)
            linestyle := :dash
            (func, a, b)
        end
    end

end

@recipe function f(ds::DataSet)
    seriestype --> :scatter
    x = ds.observables[1]
    label --> string(x.name)
    ylabel --> "events"
    markercolor --> :black
    if ds.data isa AbstractHistogram
        h = ds.data
        bins = h |> nbins
    else
        bins = get(plotattributes, :bins, 0)
        bins = bins > 0 ? bins : x.nbins > 0 ? x.nbins : 100
        h = Hist1D(ds.data, binedges=range(x.limits..., bins+1))
    end
    x := bincenters(h)
    y := bincounts(h)
    yerr := sqrt.(bincounts(h))
    # Save the attribuites for the plot
    plot_attributes[:bins] = bins
    plot_attributes[:integral] = integral(h, width=true) 
    ()
end

@recipe function f(r::FitResult)
    (; data, model) = r
    legend --> :outerleft
    legendfontsize --> 6
    labelfontsize --> 8
    legend_background_color --> :lightgray
    ylabel --> "events"
    xlabel --> string(r.model.x.name)
    @series begin
        label := "data"
        color := :black
        (data,)
    end
    @series begin
        label --> info(model)
        (model,)
    end
end

#=
@recipe function f(m::Minuit)
    isnothing(m.cost) && throw(ArgumentError("Minuit object does not have a cost function"))
    m.cost.ndim > 1 && throw(ArgumentError("Cost function dimension > 1 not supported"))
    (m.cost, m.is_valid, collect(m.values))
end

@recipe function f(cost::UnbinnedCostFunction, valid::Bool, pars::Vector{<:Real})
    nbins = get(plotattributes, :bins, 100)
    h = Hist1D(cost.data, nbins=nbins)
    edges = binedges(h)
    if valid
        @series begin
            label := info(m)
            seriestype := :path
            color := :blue
            if cost isa UnbinnedNLL
                scale = prod(Base.size(cost.data))*(edges[2]-edges[1])
                (x -> cost.model(x, pars...)*scale, first(edges), last(edges))
            else
                scale = (edges[2]-edges[1])
                (x -> cost.model(x, pars...)[2]*scale, first(edges), last(edges))
            end
        end
    end
    seriestype := :scatter
    x := bincenters(h)
    y := bincounts(h)
    yerr := sqrt.(bincounts(h))
    label := "Data"
    markercolor := :black
    ()
end
=#

end