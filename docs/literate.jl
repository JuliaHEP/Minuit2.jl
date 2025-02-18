using Literate
name = ARGS[1]

tutorialsdir =  joinpath(@__DIR__, "src/tutorials")
println("====> Processing $name")
file = "$(name)_lit.jl"

Literate.notebook(joinpath(tutorialsdir, file), tutorialsdir, name = name, execute = false, documenter = true, credit = true)
Literate.script(joinpath(tutorialsdir, file), tutorialsdir, name = name, keep_comments = false, documenter = true, credit = false)
Literate.markdown(joinpath(tutorialsdir, file), tutorialsdir, name = name, execute = true, documenter = true, credit = true)

