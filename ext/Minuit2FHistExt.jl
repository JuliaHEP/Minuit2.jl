module Minuit2FHistExt
using Minuit2
using FHist

function Minuit2.BinnedNLL(h::Hist1D, model::Function; kwargs...)
    BinnedNLL(bincounts(h), binedges(h), model; kwargs...)
end
function Minuit2.BinnedNLL(h::Hist2D, model::Function; kwargs...)
    BinnedNLL(bincounts(h), binedges(h), model; kwargs...)
end

end # module