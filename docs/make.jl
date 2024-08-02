using Documenter, Clustering

makedocs(
    source = "source",
    format = Documenter.HTML(prettyurls=false),
    sitename = "CosmoCorr.jl",
    modules = [CosmoCorr],
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
