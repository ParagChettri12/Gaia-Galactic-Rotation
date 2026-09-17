# Installs Pluto and opens the rotation-curve notebook.
# Run from the repository root with:  julia deps/build.jl
import Pkg
println("Making sure Pluto is installed")
Pkg.add(name = "Pluto", version = "0.20")
import Pluto

# Pluto installs the notebook's own packages (HTTP, CSV, LsqFit, ...) automatically
# the first time each `using` line runs, so no separate environment is needed.
notebook = joinpath(@__DIR__, "..", "Rotation.jl")
println("Opening ", notebook)
Pluto.run(notebook = notebook)
