module Minuit2PlotsExt
using Minuit2

isdefined(Base, :get_extension) ? (using Plots) : (using ..Plots)


"""
    draw_contour(m::Minuit,  x, y; bound=4, subtract_min=false, kwargs...)

Draw 2D minimal region (requires Plots module).

"""
function Minuit2.draw_contour(m::Minuit, x, y; bound=4, subtract_min=false, kwargs...)
    ix, xname = Minuit2.normalize_par(m, x)
    iy, yname = Minuit2.normalize_par(m, y)

    xv, yv, zv = Minuit2.contour(m, ix, iy, bound=bound, subtract_min=subtract_min)
    vx, vy = values(m)[ix], Minuit2.values(m)[iy]
    Plots.heatmap(xv, yv, zv', title="Contour $(m.funcname) for $(xname) vs $(yname)", xlabel=xname, ylabel=yname, kwargs...)
    Plots.scatter!([vx], [vy], label="Min", color=:white)
end

"""
    draw_mncontour(m::Minuit,  x, y; cl = 1:4, kwargs...)

Draw 2D Minos confidence region (requires Plots module).
"""
function Minuit2.draw_mncontour(m::Minuit, x, y; cl = 1:4, size=50, kwargs...)
    ix, xname = Minuit2.normalize_par(m, x)
    iy, yname = Minuit2.normalize_par(m, y)

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
    ix, xname = Minuit2.normalize_par(m, var)
    x, y = profile(m, ix; subtract_min=true, kwargs...)
    v = values(m)[ix]
    e = errors(m)[ix]
    title = text ? "$xname= $(round(v, digits=3)) - $(round(e, digits=3)) + $(round(e, digits=3))" : nothing
    plt = plot(x, y; title=title, label=m.funcname, xlabel=xname)
    vline!(plt, [v]; label=nothing, color=:black)
    band && vspan!(plt, [v - e, v + e]; label=nothing, color=:black, alpha=0.2)
    return plt
end

"""
    draw_mnprofile(m::Minuit, var; band=true, text=true, kwargs...)

Draw 1D Minos profile over a range (requires matplotlib).
"""
function Minuit2.draw_mnprofile(m::Minuit, var; band=true, text=true, kwargs...)
    ix, xname = Minuit2.normalize_par(m, var)
    x, y = mnprofile(m, ix; kwargs...)
    v = values(m)[ix]
    e = errors(m)[ix]
    title = text ? "$xname= $(round(v, digits=3)) - $(round(e, digits=3)) + $(round(e, digits=3))" : nothing
    plt = plot(x, y; title=title, label=m.funcname, xlabel=xname)
    vline!(plt, [v]; label=nothing, color=:black)
    band && vspan!(plt, [v - e, v + e]; label=nothing, color=:black, alpha=0.2)
    return plt
end

end