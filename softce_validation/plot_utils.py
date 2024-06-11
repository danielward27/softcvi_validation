import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jaxtyping import ArrayLike


def pairplot_dictionary(
    data: dict[str, ArrayLike],
    *,
    column_names: list[str] | None = None,
    filter_outliers: float = 3,
    shuffle: bool = True,
    equalize_points: bool = True,
    s: float = 10,
    limit: int = 1000,
):
    """Generate a poirplot from a dictionary of arrays.

    This plotting utility assumes that two dimensional arrays represent a set of samples
    with a batch axis, and one dimensional arrays represent a single point.

    We add special handling of one-dimensional arrays, by adding vertical lines on
    the density plots corresponding to the points, and by making the point larger.
    This is because we assume the single point has some significance (e.g. the
    ground truth, or observed value).

    Args:
        data: Dictionary of arrays.
        column_names: Column names corresponding to array columns.
        filter_outliers: Filter outliers outside the interval
            ``[Q1–filter_outliers*IQR, Q3+filter_outliers*IQR]``. Defaults to 3.
        shuffle: Wheter to shuffle points, or to overlay them in the order passed.
            Defaults to True.
        equalize_points: Whether to equalize the number of points in each 2D dataset,
            by first N points from each dataset, where N is the smallest dataset passed.
            Defaults to True.
        s: The point size.
        limit: The maximum number of points to plot from each array.
    """
    if max(arr.ndim for arr in data.values()) > 2:
        raise ValueError("Arrays must be at most 2 dimensional.")
    data_2d = {
        k: np.random.permutation(arr)[:limit]
        for k, arr in data.items()
        if arr.ndim == 2
    }
    smallest = min(arr.shape[0] for arr in data_2d.values())
    dfs = []

    if column_names is None:
        column_names = range(list(data_2d.values())[0].shape[1])

    for k, arr in data_2d.items():
        arr = _filter_outliers(arr, filter_outliers)

        if equalize_points:
            arr = arr[:smallest]

        df_i = pd.DataFrame(arr, columns=column_names)
        df_i["source"] = k
        df_i["size"] = s
        dfs.append(df_i)

    df = pd.concat(dfs)

    if shuffle:
        df = df.sample(frac=1)

    data_1d = {k: arr for k, arr in data.items() if arr.ndim == 1}

    dfs = []
    for k, arr in data_1d.items():
        arr = arr[None, :]
        df_i = pd.DataFrame(arr, columns=column_names)
        df_i["source"] = k
        df_i["size"] = 5 * s
        dfs.append(df_i)

    df = pd.concat([df, *dfs])
    df = df.reset_index(drop=True)
    sizes = df.pop("size")

    pairplot = sns.pairplot(
        df,
        hue="source",
        plot_kws={"sizes": np.array(sizes, dtype=float)},
        diag_kws={"common_norm": False},
        corner=True,
        hue_order=data.keys(),
    )

    for i, (k, arr) in enumerate(data_1d.items()):
        label_index = np.argwhere([k == data_key for data_key in data.keys()]).item()
        color = pairplot.legend.legendHandles[label_index].get_color()
        for i, ax in enumerate(np.diag(pairplot.axes)):
            y_lim = ax.get_ylim()
            ax.axvline(x=arr[i], color=color, ymin=0, ymax=y_lim[1])

    pairplot.legend.set_title(None)
    sns.move_legend(pairplot, loc=(0.6, 0.6))
    return pairplot


def _filter_outliers(data, n):
    q1 = jnp.nanpercentile(data, 25, axis=0)
    q3 = jnp.nanpercentile(data, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - n * iqr
    upper_bound = q3 + n * iqr
    mask = jnp.logical_and(data >= lower_bound, data <= upper_bound).all(axis=1)
    return data[mask]


def plot_contrastive_aux(aux, filter_quantile=0.999):
    aux_stacked = {}
    for k in aux[0].keys():
        aux_stacked[k] = jnp.stack([a[k] for a in aux])

    def filter_(arr, percentile):
        upper = jnp.quantile(arr, percentile)
        lower = jnp.quantile(arr, 1 - percentile)
        arr = arr.at[arr > upper].set(jnp.nan)
        return arr.at[arr < lower].set(jnp.nan)

    fig, axes = plt.subplots(nrows=len(aux_stacked))
    for (name, val), ax in zip(aux_stacked.items(), axes, strict=True):
        ax.plot(filter_(val, filter_quantile))
        ax.set_title(name)
    fig.tight_layout()
    plt.show()