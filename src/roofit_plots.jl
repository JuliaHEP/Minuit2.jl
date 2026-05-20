using Minuit2
using Minuit2.RooFit
using FHist, RecipesBase, Plots

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
