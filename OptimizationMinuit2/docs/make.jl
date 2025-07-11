using OptimizationMinuit2
using Documenter

DocMeta.setdocmeta!(OptimizationMinuit2, :DocTestSetup, :(using OptimizationMinuit2); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [OptimizationMinuit2],
    authors = "Moelf <proton@jling.dev>",
    repo = "https://github.com/JuliaHEP/OptimizationMinuit2.jl/blob/{commit}{path}#{line}",
    sitename = "OptimizationMinuit2.jl",
    format = Documenter.HTML(; canonical = "https://JuliaHEP.github.io/OptimizationMinuit2.jl"),
    pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/JuliaHEP/OptimizationMinuit2.jl")
