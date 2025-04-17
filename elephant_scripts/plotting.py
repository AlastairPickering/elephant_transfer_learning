"""
Functions to create UMAP plots of acoustic embeddings per call-type.
Code to plot sample spectrogram images for UMAP points adapted from https://github.com/timsainb/avgn_paper

"""

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec, lines
from scipy.ndimage import zoom
from sklearn.metrics import silhouette_samples, silhouette_score

from elephant_scripts.feature_extraction import (
    DEFAULT_SAMPLERATE,
    DEFAULT_WINDOW_SIZE,
    generate_spectrogram,
    wav_cookiecutter,
)

# Set the style of the visualisation
sns.set_theme(style="white")

# Define a colour-blind friendly palette with three colors
colorblind_palette = ["#E69F00", "#56B4E9", "#009E73"]


# Function to calculate silhouette scores and plot UMAPs
def plot_umap_with_silhouette(
    umap_df,
    model_name,
    ax,
    x="UMAP1",
    y="UMAP2",
    groupby="call_type",
):
    # Calculate silhouette scores across all data points (not per class)
    labels = umap_df["call_type"]
    silhouette_avg = silhouette_score(umap_df[[x, y]], labels)
    silhouette_values = silhouette_samples(umap_df[[x, y]], labels)

    # Prepare a dictionary to store average silhouette scores for each label
    silhouette_dict = {}
    unique_labels = labels.unique()
    for label in unique_labels:
        label_indices = labels == label
        avg_silhouette_score = silhouette_values[label_indices].mean()
        silhouette_dict[label] = avg_silhouette_score

    # Create a scatter plot with the custom colour-blind friendly palette
    scatter_plot = sns.scatterplot(
        data=umap_df,
        x=x,
        y=y,
        hue=groupby,
        palette=colorblind_palette,
        edgecolor="none",
        ax=ax,
        s=20,  # Adjusted marker size for the grid
    )

    # Set the title, labels, parameters (scaling for the grid)
    ax.set_title(
        f"{model_name} (Silhouette: {silhouette_avg:.2f})", fontsize=14
    )  # Title font size scaled down
    ax.set_xlabel(x, fontsize=12)  # Axis label font size scaled down
    ax.set_ylabel(y, fontsize=12)  # Axis label font size scaled down
    ax.tick_params(axis="x", labelsize=12)  # Tick label size scaled down
    ax.tick_params(axis="y", labelsize=12)  # Tick label size scaled down

    # Update legend labels to include silhouette scores
    handles, labels = scatter_plot.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label in silhouette_dict:
            new_label = f"{label} (Silhouette: {silhouette_dict[label]:.2f})"
            new_labels.append(new_label)
        else:
            new_labels.append(label)

    scatter_plot.legend(
        handles=handles,
        labels=new_labels,
        title=groupby.replace("_", "-").title(),
        title_fontsize=12,
        fontsize=10,
        loc="best",
    )


# Function to resize spectrograms
def resize_spectrogram(spec, target_shape=(1024, 1024), order=3):
    """
    Resizes a spectrogram to a higher resolution for sharper display.
    Args:
        spec (np.ndarray): Input spectrogram.
        target_shape (tuple): Desired output shape (height, width).
        order (int): Interpolation order for resizing (default: cubic).
    Returns:
        np.ndarray: Resized spectrogram.
    """
    zoom_factors = [t / s for t, s in zip(target_shape, spec.shape)]
    return zoom(spec, zoom_factors, order=order)


# Function to create scatter plot with spectrograms
def scatter_spec(
    umap_df,
    column_size=8,
    x="UMAP1",
    y="UMAP2",
    groupby="call_type",
    pal_color=["#E69F00", "#56B4E9", "#009E73"],
    matshow_kwargs={"cmap": "viridis"},
    scatter_kwargs={"alpha": 0.75, "s": 40},
    line_kwargs={"lw": 1, "ls": "dashed", "alpha": 0.5},
    figsize=(20, 20),
    dpi=900,
    range_pad=0.1,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    draw_lines=True,
):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(column_size, column_size)

    # Determine x range
    if x_range is None:
        xmin, xmax = np.percentile(umap_df[x], [1, 99])
        xmin -= (xmax - xmin) * range_pad
        xmax += (xmax - xmin) * range_pad
    else:
        xmin, xmax = x_range

    # Determine y range
    if y_range is None:
        ymin, ymax = np.percentile(umap_df[y], [1, 99])
        ymin -= (ymax - ymin) * range_pad
        ymax += (ymax - ymin) * range_pad
    else:
        ymin, ymax = y_range

    # Main scatter plot: include all points
    main_ax = fig.add_subplot(gs[1 : column_size - 1, 1 : column_size - 1])

    sns.scatterplot(
        data=umap_df,
        x=x,
        y=y,
        hue=groupby,
        palette=pal_color,
        edgecolor="none",
        ax=main_ax,
        legend=False,
        **scatter_kwargs,
    )

    unique_labels = list(umap_df[groupby].unique())

    # No legend in the main plot
    main_ax.set_xlabel(x, fontsize=16, labelpad=15)
    main_ax.set_ylabel(y, fontsize=16, labelpad=15)
    main_ax.set_xlim((xmin, xmax))
    main_ax.set_ylim((ymin, ymax))
    main_ax.tick_params(axis="both", labelsize=20)
    main_ax.spines["top"].set_visible(False)
    main_ax.spines["right"].set_visible(False)
    main_ax.spines["left"].set_position(("axes", 0.04))
    main_ax.spines["bottom"].set_position(("axes", 0.04))
    main_ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=6,
        width=1,
        pad=2,
        labelsize=12,
    )

    # Subsample spectrograms stratified by Call-Type
    n_subset = border_size(column_size, column_size)
    per_class = int(np.ceil(n_subset / umap_df[groupby].nunique()))
    subset = (
        umap_df.groupby(groupby)
        .sample(n=per_class, random_state=42)
        .reset_index()
        .sort_values(groupby, key=lambda x: x.map(unique_labels.index))
        .iloc[:n_subset]
    )

    col_num = subset.columns.get_loc(groupby)
    for num, annotation in enumerate(subset.itertuples()):
        # Compute the spectrogram
        wav = wav_cookiecutter(annotation)
        spec = generate_spectrogram(wav)

        # Select the cell to plot the spectrogram in
        row, col = outer_cells_clockwise(num, column_size, column_size)

        if row < 0 or col < 0:
            break

        # Plot the spectrogram
        ax = fig.add_subplot(gs[row, col])
        librosa.display.specshow(
            spec,
            ax=ax,
            cmap=matshow_kwargs.get("cmap", "magma"),
            hop_length=512,
            sr=4000,
            x_axis=None,
            y_axis=None,
        )
        ax.axis("off")

        if not draw_lines:
            continue

        # Get the position of the line end in the spectrogram
        mytrans = ax.transAxes + ax.figure.transFigure.inverted()
        line_end_pos = mytrans.transform((0.5, 0.5))

        # Get the position of the projected vocalisation in the main plot
        xpos, ypos = main_ax.transLimits.transform(
            (getattr(annotation, x), getattr(annotation, y))
        )
        mytrans2 = main_ax.transAxes + main_ax.figure.transFigure.inverted()
        infig_position_start = mytrans2.transform([xpos, ypos])

        # Draw a line connecting the spectrogram to the main plot
        group = annotation[col_num + 1]
        color = pal_color[unique_labels.index(group)]
        fig.lines.append(
            lines.Line2D(
                [infig_position_start[0], line_end_pos[0]],
                [infig_position_start[1], line_end_pos[1]],
                color=color,
                transform=fig.transFigure,
                **line_kwargs,
            )
        )

    return fig


def outer_cells_clockwise(index: int, rows: int, cols: int) -> tuple[int, int]:
    if index < cols:
        return 0, index

    if index < cols + rows - 1:
        index = index - cols
        return index + 1, cols - 1

    if index < 2 * cols + rows - 2:
        index = index - cols - rows + 2
        return rows - 1, cols - index - 1

    index = index - 2 * cols - rows + 3
    return rows - index - 1, 0


def border_size(rows: int, cols: int) -> int:
    return 2 * (rows + cols - 2)


def plot_spectrogram_matrix(
    df,
    title: str,
    window_size: float = DEFAULT_WINDOW_SIZE,
    rows: int = 3,
    cols: int = 10,
    samplerate: int = DEFAULT_SAMPLERATE,
    position: str = "middle",
):
    num_spectrograms = min(rows * cols, len(df))
    _, axes = plt.subplots(rows, cols, figsize=(cols, rows), constrained_layout=True)

    for index in range(num_spectrograms):
        annotation = df.iloc[index]
        wav = wav_cookiecutter(
            annotation,
            window_size=window_size,
            position=position,
            samplerate=samplerate,
        )
        spectrogram = generate_spectrogram(wav, samplerate)

        i = index // cols
        j = index % cols
        ax = axes[i, j]

        img = ax.imshow(
            spectrogram,
            cmap="magma",
            origin="lower",
            aspect="auto",
        )
        ax.axis("off")

    plt.suptitle(title)
    plt.colorbar(img, ax=axes, location="right", shrink=0.6)
    plt.show()
