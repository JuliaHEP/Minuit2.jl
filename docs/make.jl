using Documenter, Literate, Minuit2
 
const tutorialsdir =  joinpath(@__DIR__, "src/tutorials")
const project = @__DIR__

function process_literate(names...)
    tutorials_mds = []
    for name in names
        run(`julia --project=$project docs/literate.jl $name`)
        push!(tutorials_mds, "tutorials/$name.md")
    end
    return tutorials_mds
end

basic_mds    = process_literate("introduction")

makedocs(;
    modules=[Minuit2],
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        size_threshold = 8000000,
        size_threshold_warn = 4000000,
        repolink="https://github.com/JuliaHEP/Minuit2.jl",
    ),
    pages=[
        "Introduction" => "index.md",
        "Public API" => "api.md",
        "Tutorials" => [ "Basic" => basic_mds ],
        "Release Notes" => "release_notes.md",
    ],
    checkdocs=:exports,
    repo="https://github.com/JuliaHEP/Minuit2.jl/blob/{commit}{path}#L{line}",
    sitename="Minuit2.jl",
    authors="Pere Mato",
)

deploydocs(;
    repo="github.com/JuliaHEP/Minuit2.jl",
    push_preview = true
)
