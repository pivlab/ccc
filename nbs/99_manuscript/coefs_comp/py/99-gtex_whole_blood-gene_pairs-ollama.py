# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# It selects a set of specific gene pairs from a tissue, and checks if relationship are replicated on other tissues.
# It also uses GTEx metadata (such as sex) to explain relationships.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd

from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from ccc import conf
from ccc.coef import ccc

# %% [markdown] tags=[]
# # Settings

# %%
# this gene pair was originally found with ccc on whole blood
# interesting: https://clincancerres.aacrjournals.org/content/26/21/5567.figures-only
gene0_id, gene1_id = "ENSG00000147050.14", "ENSG00000183878.15"
gene0_symbol, gene1_symbol = "KDM6A", "UTY"

# %% [markdown] tags=[]
# # Paths

# %%
TISSUE_DIR = conf.GTEX["DATA_DIR"] / "data_by_tissue"
assert TISSUE_DIR.exists()

# %% tags=[]
OUTPUT_FIGURE_DIR = (
    conf.MANUSCRIPT["FIGURES_DIR"]
    / "coefs_comp"
    / f"{gene0_symbol.lower()}_vs_{gene1_symbol.lower()}"
)
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% [markdown]
# # Data

# %% [markdown] tags=[]
# ## GTEx metadata

# %%
gtex_metadata = pd.read_pickle(conf.GTEX["DATA_DIR"] / "gtex_v8-sample_metadata.pkl")

# %%
gtex_metadata.shape

# %%
gtex_metadata.head()

# %% [markdown] tags=[]
# ## Gene Ensembl ID -> Symbol mapping

# %%
gene_map = pd.read_pickle(conf.GTEX["DATA_DIR"] / "gtex_gene_id_symbol_mappings.pkl")

# %%
gene_map = gene_map.set_index("gene_ens_id")["gene_symbol"].to_dict()

# %%
assert gene_map["ENSG00000145309.5"] == "CABS1"

# %%
assert gene_map[gene0_id] == gene0_symbol
assert gene_map[gene1_id] == gene1_symbol


# %% [markdown]
# # Plot

# %%
def get_tissue_file(name):
    """
    Given a part of a tissue name, it returns a file path to the
    expression data for that tissue in GTEx. It fails if more than
    one files are found.

    Args:
        name: a string with the tissue name (or a part of it).

    Returns:
        A Path object pointing to the gene expression file for the
        given tissue.
    """
    tissue_files = []
    for f in TISSUE_DIR.glob("*.pkl"):
        if name in f.name:
            tissue_files.append(f)

    assert len(tissue_files) == 1
    return tissue_files[0]


# %%
# testing
_tmp = get_tissue_file("whole_blood")
assert _tmp.exists()


# %%
def simplify_tissue_name(tissue_name):
    return f"{tissue_name[0].upper()}{tissue_name[1:].replace('_', ' ')}"


# %%
assert simplify_tissue_name("whole_blood") == "Whole blood"
assert simplify_tissue_name("uterus") == "Uterus"


# %%
def plot_gene_pair(
    tissue_name, gene0, gene1, hue=None, kind="hex", ylim=None, bins="log"
):
    """
    It plots (joint plot) a gene pair from the given tissue. It saves the plot
    for the manuscript.
    """
    # merge gene expression with metadata
    tissue_file = get_tissue_file(tissue_name)
    tissue_data = pd.read_pickle(tissue_file).T[[gene0, gene1]]
    tissue_data = pd.merge(
        tissue_data,
        gtex_metadata,
        how="inner",
        left_index=True,
        right_index=True,
        validate="one_to_one",
    )

    # get gene symbols
    gene0_symbol, gene1_symbol = gene_map[gene0], gene_map[gene1]
    display((gene0_symbol, gene1_symbol))

    # compute correlations for this gene pair
    _clustermatch = ccc(tissue_data[gene0], tissue_data[gene1])
    _pearson = pearsonr(tissue_data[gene0], tissue_data[gene1])[0]
    _spearman = spearmanr(tissue_data[gene0], tissue_data[gene1])[0]

    _title = f"{simplify_tissue_name(tissue_name)}\n$c={_clustermatch:.2f}$  $p={_pearson:.2f}$  $s={_spearman:.2f}$"

    other_args = {
        "kind": kind,  # if hue is None else "scatter",
        "rasterized": True,
    }
    if hue is None:
        # other_args["bins"] = bins
        pass
    else:
        other_args["hue_order"] = ["Male", "Female"]

    with sns.plotting_context("paper", font_scale=1.5):
        p = sns.jointplot(
            data=tissue_data,
            x=gene0,
            y=gene1,
            hue=hue,
            **other_args,
            # ylim=(0, 500),
        )

        if ylim is not None:
            p.ax_joint.set_ylim(ylim)

        gene_x_id = p.ax_joint.get_xlabel()
        gene_x_symbol = gene_map[gene_x_id]
        p.ax_joint.set_xlabel(f"{gene_x_symbol}", fontstyle="italic")

        gene_y_id = p.ax_joint.get_ylabel()
        gene_y_symbol = gene_map[gene_y_id]
        p.ax_joint.set_ylabel(f"{gene_y_symbol}", fontstyle="italic")

        p.fig.suptitle(_title)

        # save
        output_file = (
            OUTPUT_FIGURE_DIR
            / f"gtex_{tissue_name}-{gene_x_symbol}_vs_{gene_y_symbol}.png"
        )
        # display(output_file)

        plt.savefig(
            output_file,
            bbox_inches="tight",
            dpi=300,
            facecolor="white",
        )
        
        # display(p)

    return p, output_file


# %% [markdown]
# ## Read Haoyu gene pairs

# %%
list(conf.DATA_DIR.glob("*"))

# %%
hgp = pd.read_csv(conf.DATA_DIR / "haoyu_gene_pairs.txt", sep="\s+")

# %%
hgp.columns

# %%
hgp.head()

# %%
i = 0
gene0_id, gene1_id = hgp.iloc[i,:2]
gene0_id = gene0_id.split("'")[1]
gene1_id = gene1_id.split("'")[1]
display(gene0_id, gene1_id)

# %%

# %%
from ollama import Client
client = Client(
  host='http://host.docker.internal:11434',
)

def analyze_figure(output_file):
    response = client.chat(
        # model='gemma3:27b',
        model='qwen2.5vl:32b',
        # model='qwen2.5vl:7b',
        messages=[
      {
        'role': 'user',
        'content':
        """
Analyze the image, which shows a scatter plot with a pattern between two genes.
Classify the pattern in one of these categories:
1. Two lines, one with positive slope and the other with negative slope.
2. Two lines, both with positive slope.
3. Other.
Respond only with the category name. Do not explain anything.
        """.strip(),
        'images': [output_file],
      },
    ])

    # print(response['message']['content'])
    # or access fields directly from the response object
    print(response.message.content)
    
    return response.message.content.strip()


# %%
# YES
analyze_figure(OUTPUT_FIGURE_DIR / "test" / "pattern0.png")
analyze_figure(OUTPUT_FIGURE_DIR / "test" / "pattern1.png")

# %%
# NO
analyze_figure(OUTPUT_FIGURE_DIR / "test" / "pattern2.png")
analyze_figure(OUTPUT_FIGURE_DIR / "test" / "pattern3.png")
analyze_figure(OUTPUT_FIGURE_DIR / "test" / "pattern4.png")

# %%

# %%
for i in range(0, 10):
    gene0_id, gene1_id = hgp.iloc[i,:2]
    gene0_id = gene0_id.split("'")[1]
    gene1_id = gene1_id.split("'")[1]
    # display(gene0_id, gene1_id)
    
    p, output_file = plot_gene_pair(
        "whole_blood",
        gene0_id,
        gene1_id,
        # hue="SEX",
        kind="scatter",
    )
    
    display(p)
    
    analyze_figure(output_file)

# %%

# %%

# %%
