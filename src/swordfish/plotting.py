import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_stats(df: pd.DataFrame, plot_type: int) -> None:
    """
    Plot the results of the experiments.
    :param df: pd.DataFrame of results.
    :param plot_type: experiment type.
    """
    name = (
        "swordfish_all_opts_comparison"
        if plot_type == 1
        else "swordfish_all_types_comparison"
    )
    exp_range = np.sort(np.log2(df["n"].unique()).astype(int))
    # remap the algorithm names and plot_types to work better in the legend
    df["algorithm"] = df["algorithm"].map(
        lambda nm: "none" if nm == "Swordfish" else f"$\\texttt{{{nm[10:-1]}}}$"
    )
    df["problem_type"] = df["problem_type"].map(lambda nm: f"$\\texttt{{{nm}}}$")

    sns.set_theme(
        style="whitegrid",
        font_scale=1.8,
        context="paper",
        font="serif",
        rc={
            "text.usetex": True,
            "lines.linewidth": 0.5,
        },
    )
    if plot_type == 1:
        g = sns.FacetGrid(
            df,
            row="cost_type",
            col="problem_type",
            sharex=False,
            height=5,
            aspect=1.7,
            despine=True,
            legend_out=False,
        )
    else:
        g = sns.FacetGrid(
            df,
            row="problem_type",
            col="cost_type",
            height=4,
            sharex=False,
            aspect=1.8,
            despine=True,
            legend_out=False,
        )

    g.map_dataframe(
        sns.boxenplot,
        x="n",
        y="normalized_cost",
        hue="algorithm",
        orient="v",
        dodge=True,
        palette=sns.color_palette("colorblind")[: df["algorithm"].nunique()],
        scale="linear",
        showfliers=False,
    )
    g.tick_params(labelsize=18)
    if plot_type == 1:
        g.set_titles(template="{row_name} - {col_name}", pad=40, size=26)
    else:
        g.set_titles(template="{col_name}", pad=20, size=22)

    g.set_xticklabels(
        labels=list(map(lambda x: f"$2^{{{x}}}$", exp_range)), fontsize=22
    )
    g.set_xlabels(
        r"$n$",
        fontsize=28,
    )
    g.set_ylabels(
        r"$|Q| / (\frac{3}{2}n\log_2 n)$",
        fontsize=22,
        loc="top",
        rotation="horizontal",
    )
    for ax in g.axes[:, 0]:
        ax.get_yaxis().set_label_coords(0.2, 1.01)

    def annotate(data, **_):
        ptype = data["problem_type"].iloc[0]
        ax1 = plt.gca()
        ax1.text(
            0.35,
            0.85,
            ptype,
            transform=ax1.transAxes,
            fontsize=20,
            bbox=dict(facecolor="none", edgecolor="gray", boxstyle="round,pad=0.3"),
        )

    for ix, r_axes in enumerate(g.axes):
        for ax in r_axes:
            if plot_type == 1:
                ax.set_xlabel(r"$n$", fontsize=24)
            if plot_type == 2 and ix > 0:
                ax.set_ylabel("")
                ax.set_title("")
    if plot_type == 2:
        g.map_dataframe(annotate)

    g.refline(y=1, color="gray", linestyle="--", linewidth=1.5)
    g.add_legend(title="optimization", fontsize=15, framealpha=1, loc="upper right")
    plt.setp(g._legend.get_title(), fontsize=14)

    g.tight_layout()
    plt.savefig(f"./{name}.svg", bbox_inches="tight")


if __name__ == "__main__":
    for name_, pt in (
        ("swordfish_all_opts_comparison", 1),
        ("swordfish_all_types_comparison", 2),
    ):
        plot_stats(pd.read_csv(f"../{name_}.csv"), pt)
