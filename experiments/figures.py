import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# These are based on the JMLR template.
FULL_WIDTH = 6.0
HALF_WIDTH = FULL_WIDTH / 2 - 0.1
THIRD_WIDTH = FULL_WIDTH / 3 - 0.1
COMPACT_HEIGHT = 1.8
MEDIUM_HEIGHT = 2.5
LARGER_FONT_SIZE = 10
SMALLER_FONT_SIZE = 9

MMD_COLOR = "C1"
KSD_COLOR = "C0"
WILD_COLOR = "C2"
PARAMETRIC_COLOR = "C3"
ALT_COLOR = "C4"

squashed_legend_params = {
    "handlelength": 1.0,
    "handletextpad": 0.5,
    "labelspacing": 0.3,
    "borderaxespad": 0.2,
    "borderpad": 0.25,
    "columnspacing": 0.7,
}
squashed_label_params = {"labelpad": 1.5}


def configure_matplotlib() -> None:
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amsfonts}")
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = "10"
    matplotlib.rcParams["font.size"] = LARGER_FONT_SIZE
    matplotlib.rcParams["figure.figsize"] = [FULL_WIDTH * 0.9, MEDIUM_HEIGHT]


def save_fig(name: str, **tight_layout_kwargs) -> None:
    plt.tight_layout(**tight_layout_kwargs)
    plt.savefig(f"figures/{name}.png", dpi=200)
    plt.savefig(f"figures/{name}.pdf", bbox_inches="tight", transparent=True)


def set_axis_color(ax: Axes, color: str) -> None:
    ax.spines["left"].set_color(color)
    ax.spines["right"].set_color(color)
    ax.tick_params(axis="y", colors=color)
    ax.yaxis.label.set_color(color)
