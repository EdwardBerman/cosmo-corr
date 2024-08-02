using Documenter

makedocs(
    source = "source",
    format = Documenter.HTML(prettyurls=false),
    sitename = "CosmoCorr.jl",
    pages = [
        "Introduction" => "index.md",
        "Algorithms" => [
            "naive.md",
            "treecorr.md",
            "hclust.md",
        ],
    ],
)

deploydocs(
    repo = "https://github.com/EdwardBerman/CosmoCorr.git",
)

# specify under sitename when the module is published in pkg: modules = [CosmoCorr],
