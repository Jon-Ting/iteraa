"""Plotting utilities for archetypal analysis.

This module provides functions for visualizing data and archetypes in various
spaces, such as archetype space (simplex, radar plots) and feature space
(parallel coordinates, feature profiles).
"""

from os import mkdir
from os.path import exists
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE

from iteraa.constants import DPI, FIGS_DIR_PATH, NUM_JOBS, PALETTE, RANDOM_STATE
from iteraa.utils import ecdf


def _ensure_figs_dir():
    """Ensures the directory for figures exists."""
    if not exists(FIGS_DIR_PATH):
        mkdir(FIGS_DIR_PATH)


def plotRadarDatapoints(
    AA: Any,
    X: np.ndarray,
    sampIDs: list[int] = [0],
    archSpaceIDs: list[int] = [0, 1],
    sepSamps: bool = False,
    showLabel: bool = True,
    labelAll: bool = False,
    showLegend: bool = False,
    figSize: tuple[int, int] = (6, 6),
    dpi: int = DPI,
    title: str | None = None,
    figNamePrefix: str = "",
):
    """Visualizes data points in the archetype space using a radar plot.

    Args:
        AA: Fitted ArchetypalAnalysis object.
        X: The dataset.
        sampIDs: Indices of samples to plot.
        archSpaceIDs: Indices of archetypes to include in the radar plot.
        sepSamps: Whether to save each sample in a separate plot.
        showLabel: Whether to show labels on the radar axes.
        labelAll: Whether to label all axes (if False, labels only standard corners).
        showLegend: Whether to show a legend.
        figSize: Figure size.
        dpi: DPI for the figure.
        title: Figure title.
        figNamePrefix: Prefix for the saved figure filename.
    """
    labels = [f"A{i + 1}" for i in archSpaceIDs]
    angles = np.linspace(0, 2 * np.pi, len(archSpaceIDs), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    if not sepSamps:
        fig = plt.figure(figsize=figSize, dpi=dpi)
        ax = fig.add_subplot(111, polar=True)
        if showLegend:
            legend = []

    for i, sampID in enumerate(sampIDs):
        if sepSamps:
            fig = plt.figure(figsize=figSize, dpi=dpi)
            ax = fig.add_subplot(111, polar=True)

        c = sns.color_palette("husl", len(sampIDs))[i]
        _, alfaX = AA.transform(X[sampID, :].reshape(1, -1))
        alfaX_subset = [alfaX[ID, 0] for ID in archSpaceIDs]
        alfaX_subset = np.concatenate((alfaX_subset, [alfaX_subset[0]]))

        ax.plot(angles, alfaX_subset, ".-", linewidth=1, markersize=5, zorder=-1, color=c)
        ax.fill(angles, alfaX_subset, alpha=0.2, zorder=-2, color=c)
        ax.set_rticks([0.2, 0.4, 0.6, 0.8])
        ax.set_rlim(0.0, 1.0)

        if showLabel:
            if not labelAll and len(archSpaceIDs) < 8:
                labelAll = True
            if labelAll:
                labelIdxs = list(range(len(angles) - 1))
            else:
                labelIdxs = np.round(np.linspace(0, len(angles) - 1, 9)[:-1]).astype(int)
            ax.set_thetagrids(angles[labelIdxs] * 180.0 / np.pi, [labels[idx] for idx in labelIdxs])
        else:
            ax.set_thetagrids(angles * 180.0 / np.pi, [""] * len(angles))

        if title:
            ax.set_title(title)
        ax.grid(linestyle="dotted", color="k", alpha=0.3, linewidth=1)
        ax.set_facecolor("#EAECEE")

        if not sepSamps:
            if showLegend:
                legend.extend([f"D{sampID + 1}", "_"])
        else:
            _ensure_figs_dir()
            plt.savefig(
                f"{FIGS_DIR_PATH}/{figNamePrefix}_sampsArchSpace_D{sampID + 1}.png",
                bbox_inches="tight",
            )

    if not sepSamps:
        if showLegend:
            ax.legend(legend, loc="center", bbox_to_anchor=(1.2, 0.5), ncol=1)
        _ensure_figs_dir()
        plt.savefig(f"{FIGS_DIR_PATH}/{figNamePrefix}_sampsArchSpace.png", bbox_inches="tight")


def createSimplexAx(
    AA: Any,
    archIDs: list[int] = [0, 1, 2],
    gridOn: bool = True,
    showLabel: bool = True,
    labelAll: bool = False,
    figSize: tuple[int, int] = (3, 3),
    gridLineWidth: float = 0.5,
    gridcolor: str = "k",
    bordercolor: str = "k",
    fontcolor: str = "k",
) -> plt.Figure:
    """Creates a simplex (triangle/polygon) axis for visualizing data in archetype space.

    Args:
        AA: Fitted ArchetypalAnalysis object.
        archIDs: Indices of archetypes to form the simplex vertices.
        gridOn: Whether to show grid lines.
        showLabel: Whether to show labels on vertices.
        labelAll: Whether to label all vertices.
        figSize: Figure size.
        gridLineWidth: Width of grid lines.
        gridcolor: Color of grid lines.
        bordercolor: Color of the simplex border.
        fontcolor: Color of labels.

    Returns:
        The matplotlib Figure object.
    """
    if len(archIDs) == 0:
        raise ValueError("Archetype IDs can't be empty!")

    fig, ax = plt.subplots(figsize=figSize)

    labels = ["A" + str(i + 1) for i in archIDs] if showLabel else []
    labelOffset = 0.10
    sides = len(archIDs)

    basis = np.array(
        [
            [np.cos(2 * i * np.pi / sides + np.pi / 2), np.sin(2 * i * np.pi / sides + np.pi / 2)]
            for i in range(sides)
        ]
    )

    if not labelAll and len(labels) < 8:
        labelAll = True

    if showLabel:
        labelIdxs = (
            np.round(np.linspace(0, len(labels) - 1, 9)[:-1]).astype(int)
            if not labelAll
            else range(len(labels))
        )
        for i, l in enumerate(labels):
            if not labelAll and i not in labelIdxs:
                continue
            if i >= sides:
                break
            x, y = basis[i, 0], basis[i, 1]
            angle = 180 * np.arctan2(y, x) / np.pi + 90
            if angle > 90 and angle <= 270:
                angle = (angle + 180) % 360
            ax.text(
                x * (1 + labelOffset),
                y * (1 + labelOffset),
                l,
                horizontalalignment="center",
                verticalalignment="center",
                rotation=angle,
                fontsize=12,
                color=fontcolor,
            )

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_frame_on(False)

    if gridOn:
        lstAx0, lstAx1 = [], []
        for i in range(sides):
            for j in range(i + 2, sides):
                if not (i == 0 and j == sides - 1):
                    lstAx0.extend([basis[i, 0], 0, basis[j, 0], 0])
                    lstAx1.extend([basis[i, 1], 0, basis[j, 1], 0])
        ax.plot(lstAx0, lstAx1, color=gridcolor, linewidth=gridLineWidth, alpha=0.5, zorder=1)

    border = np.vstack([basis, basis[0, :]])
    ax.plot(border[:, 0], border[:, 1], linewidth=1.5, color=bordercolor, zorder=2)

    return fig


def mapAlfaToSimplex(alfa: np.ndarray, AA: Any) -> np.ndarray:
    """Maps the alfa matrix to a 2D simplex representation.

    Args:
        alfa: The alfa matrix (nArchetypes, nData).
        AA: Fitted ArchetypalAnalysis object.

    Returns:
        Mapped coordinates. Shape: (nData, 2).
    """
    nArchetypes = AA.nArchetypes
    basis = np.array(
        [
            [
                np.cos(2 * i * np.pi / nArchetypes + np.pi / 2),
                np.sin(2 * i * np.pi / nArchetypes + np.pi / 2),
            ]
            for i in range(nArchetypes)
        ]
    )
    return np.dot(alfa.T, basis)


def plotSimplex(
    AA: Any,
    alfa: np.ndarray,
    archIDs: list[int] = [0, 1, 2],
    plotArgs: dict[str, Any] = {},
    gridOn: bool = True,
    showLabel: bool = True,
    labelAll: bool = False,
    figSize: tuple[int, int] = (3, 3),
    dpi: int = DPI,
    gridLineWidth: float = 0.1,
    color: str = "#303F9F",
    alpha: float = 0.8,
    markerSize: int = 20,
    figNamePrefix: str = "",
):
    """Visualizes the dataset in a simplex formed by archetypes.

    Args:
        AA: Fitted ArchetypalAnalysis object.
        alfa: The alfa matrix.
        archIDs: Indices of archetypes to form the simplex vertices.
        plotArgs: Additional keyword arguments for scatter plot.
        gridOn: Whether to show grid lines.
        showLabel: Whether to show labels on vertices.
        labelAll: Whether to label all vertices.
        figSize: Figure size.
        dpi: DPI for the figure.
        gridLineWidth: Width of grid lines.
        color: Default color for data points.
        alpha: Alpha value for data points.
        markerSize: Size of data points.
        figNamePrefix: Prefix for the saved figure filename.
    """
    if len(archIDs) == 0:
        raise ValueError("Archetype IDs can't be empty!")

    labels = ["A" + str(i + 1) for i in archIDs] if showLabel else []
    labelOffset = 0.10
    data = alfa[archIDs].T
    sides = len(archIDs)

    basis = np.array(
        [
            [np.cos(2 * i * np.pi / sides + np.pi / 2), np.sin(2 * i * np.pi / sides + np.pi / 2)]
            for i in range(sides)
        ]
    )

    newdata = np.dot(data, basis)
    fig = plt.figure(figsize=figSize, dpi=dpi)
    ax = fig.add_subplot(111)

    if not labelAll and len(labels) < 8:
        labelAll = True

    if showLabel:
        labelIdxs = (
            np.round(np.linspace(0, len(labels) - 1, 9)[:-1]).astype(int)
            if not labelAll
            else range(len(labels))
        )
        for i, l in enumerate(labels):
            if not labelAll and i not in labelIdxs:
                continue
            if i >= sides:
                break
            x, y = basis[i, 0], basis[i, 1]
            angle = 180 * np.arctan2(y, x) / np.pi + 90
            if angle > 90 and angle <= 270:
                angle = (angle + 180) % 360
            ax.text(
                x * (1.05 + labelOffset),
                y * (1.05 + labelOffset),
                l,
                horizontalalignment="center",
                verticalalignment="center",
                rotation=angle,
            )

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_frame_on(False)

    if gridOn:
        lstAx0, lstAx1 = [], []
        for i in range(sides):
            for j in range(i + 2, sides):
                if not (i == 0 and j == sides - 1):
                    lstAx0.extend([basis[i, 0], 0, basis[j, 0], 0])
                    lstAx1.extend([basis[i, 1], 0, basis[j, 1], 0])
        ax.plot(lstAx0, lstAx1, color="#212121", linewidth=gridLineWidth, alpha=0.5, zorder=1)

    border = np.vstack([basis, basis[0, :]])
    ax.plot(border[:, 0], border[:, 1], linewidth=1, zorder=2, color="k")

    if not plotArgs:
        ax.scatter(
            newdata[:, 0],
            newdata[:, 1],
            color=color,
            zorder=3,
            alpha=alpha,
            s=markerSize,
            edgecolor="k",
            linewidth=0.8,
        )
    else:
        if "marker" in plotArgs:
            markerVals = plotArgs["marker"].values
            markerUnq = np.unique(markerVals)
            for marker in markerUnq:
                rowIdx = np.where(markerVals == marker)
                tmpArg = {k: v.values[rowIdx] for k, v in plotArgs.items() if k != "marker"}
                ax.scatter(
                    newdata[rowIdx, 0],
                    newdata[rowIdx, 1],
                    **tmpArg,
                    marker=marker,
                    edgecolor="k",
                    alpha=alpha,
                    zorder=3,
                )
        else:
            ax.scatter(newdata[:, 0], newdata[:, 1], **plotArgs, marker="s", zorder=3, alpha=alpha)

    _ensure_figs_dir()
    plt.savefig(f"{FIGS_DIR_PATH}/{figNamePrefix}_simplex.png", bbox_inches="tight")


def parallelPlot(
    AA: Any,
    lstFeat: list[str],
    dfColor: pd.DataFrame,
    featIDs: list[int] = [0, 1, 2],
    archIDs: list[int] = [0, 1, 2],
    sampIDs: list[int] = [0, 1, 2],
    linewidth: float = 0.3,
    archColor: str = "k",
    figSize: tuple[int, int] = (15, 5),
    dpi: int = DPI,
    figNamePrefix: str = "",
):
    """Visualizes the dataset and archetypes using parallel coordinates.

    Args:
        AA: Fitted ArchetypalAnalysis object.
        lstFeat: List of all feature names.
        dfColor: Dataframe containing a 'color' column for each sample.
        featIDs: Indices of features to plot.
        archIDs: Indices of archetypes to plot.
        sampIDs: Indices of samples to plot.
        linewidth: Line width for data points.
        archColor: Line color for archetypes.
        figSize: Figure size.
        dpi: DPI for the figure.
        figNamePrefix: Prefix for the saved figure filename.
    """
    from matplotlib import ticker

    selected_feat_names = [lstFeat[i] for i in featIDs]
    x = list(range(len(selected_feat_names)))

    # Prepare data for plotting
    data_points = AA.X[featIDs, :][:, sampIDs].T
    df = pd.DataFrame(data_points, columns=selected_feat_names)

    colors = dfColor.iloc[sampIDs].copy()

    for i in archIDs:
        arch_data = AA.archetypes_[featIDs, i]
        df.loc[len(df)] = arch_data
        colors.loc[len(colors)] = {"color": "arch"}

    # Normalize data for each column
    minMaxRange = {}
    df_norm = df.copy()
    for col in selected_feat_names:
        minVal, maxVal = df[col].min(), df[col].max()
        valRange = max(maxVal - minVal, 1e-9)
        minMaxRange[col] = [minVal, maxVal, valRange]
        df_norm[col] = (df[col] - minVal) / valRange

    fig, axes = plt.subplots(1, len(x) - 1, sharey=False, figsize=figSize, dpi=dpi)

    for i, ax in enumerate(axes):
        for idx in df_norm.index:
            if colors.iloc[idx]["color"] == "arch":
                ax.plot(x, df_norm.iloc[idx], color=archColor, alpha=0.8, linewidth=2.0)
            else:
                ax.plot(
                    x,
                    df_norm.iloc[idx],
                    color=colors.iloc[idx]["color"],
                    alpha=0.3,
                    linewidth=linewidth,
                )
        ax.set_xlim([x[i], x[i + 1]])

    def set_ticks_for_axis(dim, ax, ticks):
        col = selected_feat_names[dim]
        minVal, _, valRange = minMaxRange[col]
        step = valRange / (ticks - 1)
        tickLabels = [round(minVal + step * i, 2) for i in range(ticks)]
        normStep = 1.0 / (ticks - 1)
        ticks_pos = [normStep * i for i in range(ticks)]
        ax.yaxis.set_ticks(ticks_pos)
        ax.set_yticklabels(tickLabels)

    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax, ticks=6)
        ax.set_xticklabels([selected_feat_names[dim]], rotation="vertical")

    # Final axis
    ax = axes[-1]
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(len(axes), ax, ticks=6)
    ax.set_xticklabels([selected_feat_names[-2], selected_feat_names[-1]], rotation="vertical")

    plt.subplots_adjust(wspace=0)
    _ensure_figs_dir()
    plt.savefig(f"{FIGS_DIR_PATH}/{figNamePrefix}_parallel.png", bbox_inches="tight")


def plotProfile(
    AA: Any,
    allFeatNames: list[str] | None = None,
    featIDs: list[int] = [0],
    archIDs: list[int] = [0, 1],
    figSize: tuple[int, int] = (14, 5),
    dpi: int = DPI,
    figNamePrefix: str = "",
):
    """Plots the feature profile of archetypes.

    Args:
        AA: Fitted ArchetypalAnalysis object.
        allFeatNames: List of all feature names.
        featIDs: Indices of features to plot.
        archIDs: Indices of archetypes to plot.
        figSize: Figure size.
        dpi: DPI for the figure.
        figNamePrefix: Prefix for the saved figure filename.
    """
    if not archIDs:
        raise ValueError("Archetype IDs can't be empty!")

    if not hasattr(AA, "archetypeProfile") or AA.archetypeProfile is None:
        # Compute profiles if not present
        AA.archetypeProfile = np.zeros([AA.nDim, AA.nArchetypes])
        for i in range(AA.nArchetypes):
            for j in range(AA.nDim):
                AA.archetypeProfile[j, i] = ecdf(AA.X[j, :], AA.archetypes_[j, i])

    sns.set_style("ticks")
    xVals = np.arange(1, len(featIDs) + 1)
    for i in archIDs:
        plt.figure(figsize=figSize, dpi=dpi)
        plt.bar(
            xVals,
            AA.archetypeProfile[featIDs, i] * 100.0,
            color="#D32F2F",
            edgecolor="#212121",
            linewidth=0.8,
        )
        if allFeatNames:
            plt.xticks(xVals, [allFeatNames[idx] for idx in featIDs], rotation="vertical")
        plt.ylim([0, 100])
        plt.ylabel(f"A{i + 1}")
        plt.grid(linestyle="dotted")
        plt.tight_layout()
        _ensure_figs_dir()
        plt.savefig(f"{FIGS_DIR_PATH}/{figNamePrefix}_A{i + 1}_featProf.png", bbox_inches="tight")


def plotRadarProfile(
    AA: Any,
    allFeatNames: list[str] | None = None,
    featIDs: list[int] = [0],
    archIDs: list[int] = [0, 1],
    fillAlpha: float = 0.2,
    linewidth: float = 1.0,
    ncol: int = 1,
    sepArchs: bool = False,
    showLabel: bool = True,
    labelAll: bool = False,
    showName: bool = False,
    closeFig: bool = False,
    showLegend: bool = True,
    figSize: tuple[int, int] = (6, 6),
    dpi: int = DPI,
    title: str | None = None,
    figNamePrefix: str = "",
):
    """Visualizes archetype feature profiles using a radar plot.

    Args:
        AA: Fitted ArchetypalAnalysis object.
        allFeatNames: List of all feature names.
        featIDs: Indices of features to include.
        archIDs: Indices of archetypes to include.
        fillAlpha: Alpha value for fill color.
        linewidth: Line width.
        ncol: Number of columns in legend.
        sepArchs: Whether to save each archetype in a separate plot.
        showLabel: Whether to show labels on axes.
        labelAll: Whether to label all axes.
        showName: Whether to show feature names instead of indices.
        closeFig: Whether to close the figure after saving.
        showLegend: Whether to show a legend.
        figSize: Figure size.
        dpi: DPI for the figure.
        title: Figure title.
        figNamePrefix: Prefix for the saved figure filename.
    """
    if not archIDs:
        raise ValueError("Archetype IDs can't be empty!")
    if not featIDs:
        raise ValueError("No feature is selected!")

    if not hasattr(AA, "archetypeProfile") or AA.archetypeProfile is None:
        # Compute profiles if not present
        AA.archetypeProfile = np.zeros([AA.nDim, AA.nArchetypes])
        for i in range(AA.nArchetypes):
            for j in range(AA.nDim):
                AA.archetypeProfile[j, i] = ecdf(AA.X[j, :], AA.archetypes_[j, i])

    labels = [allFeatNames[i] if showName and allFeatNames else f"F{i}" for i in featIDs]
    angles = np.linspace(0, 2 * np.pi, len(featIDs), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    if not sepArchs:
        fig = plt.figure(figsize=figSize, dpi=dpi)
        ax = fig.add_subplot(111, polar=True)
        if showLegend:
            legend = []

    for i, archID in enumerate(archIDs):
        if sepArchs:
            fig = plt.figure(figsize=figSize, dpi=dpi)
            ax = fig.add_subplot(111, polar=True)

        c = sns.color_palette("husl", len(archIDs))[i]
        xProfile = AA.archetypeProfile[featIDs, archID]
        xProfile = np.concatenate((xProfile, [xProfile[0]]))

        ax.plot(angles, xProfile, ".-", linewidth=linewidth, markersize=5, zorder=-1, color=c)
        ax.fill(angles, xProfile, alpha=fillAlpha, zorder=-2, color=c)
        ax.set_rticks([0.2, 0.4, 0.6, 0.8])
        ax.set_rlim(0.0, 1.0)

        if showLabel:
            labelIdxs = (
                list(range(len(angles) - 1))
                if labelAll
                else np.round(np.linspace(0, len(angles) - 1, 9)[:-1]).astype(int)
            )
            ax.set_thetagrids(angles[labelIdxs] * 180.0 / np.pi, [labels[idx] for idx in labelIdxs])
        else:
            ax.set_thetagrids(angles * 180.0 / np.pi, [""] * len(angles))

        if title:
            ax.set_title(title)
        ax.grid(linestyle="dotted", color="k", alpha=0.3, linewidth=1)
        ax.set_facecolor("#EAECEE")

        if not sepArchs:
            if showLegend:
                legend.extend([f"A{archID + 1}", "_"])
        else:
            _ensure_figs_dir()
            plt.savefig(
                f"{FIGS_DIR_PATH}/{figNamePrefix}_radarProf_A{archID + 1}.png", bbox_inches="tight"
            )
            if closeFig:
                plt.close()

    if not sepArchs:
        if showLegend:
            ax.legend(legend, loc="center", bbox_to_anchor=(0.5, -0.3), ncol=ncol)
        _ensure_figs_dir()
        plt.savefig(f"{FIGS_DIR_PATH}/{figNamePrefix}_radarProf.png", bbox_inches="tight")


def plotCloseMatch(
    AA: Any,
    archIDs: list[int] = [0, 1],
    archSpaceIDs: list[int] = [0, 1],
    sepSamps: bool = False,
    showLabel: bool = True,
    labelAll: bool = False,
    showLegend: bool = False,
    figSize: tuple[int, int] = (6, 6),
    dpi: int = DPI,
    title: str | None = None,
    figNamePrefix: str = "",
):
    """Visualizes the samples closest to archetypes in the archetype space.

    Args:
        AA: Fitted ArchetypalAnalysis object.
        archIDs: Indices of archetypes to find matches for.
        archSpaceIDs: Indices of archetypes to include in the radar plot.
        sepSamps: Whether to save each match in a separate plot.
        showLabel: Whether to show labels on the radar axes.
        labelAll: Whether to label all axes.
        showLegend: Whether to show a legend.
        figSize: Figure size.
        dpi: DPI for the figure.
        title: Figure title.
        figNamePrefix: Prefix for the saved figure filename.
    """
    if not archIDs or not archSpaceIDs:
        raise ValueError("Archetype IDs can't be empty!")

    # Extract most archetypal samples if not done
    if not hasattr(AA, "closeMatch") or not AA.closeMatch:
        AA.closeMatch = {}
        for i in range(AA.nArchetypes):
            r = cdist(AA.X.T, AA.archetypes_[:, i].reshape(1, AA.nDim))
            iMin = np.argmin(r)
            AA.closeMatch[i + 1] = (iMin, AA.alfa_[:, iMin])

    labels = [f"A{i + 1}" for i in archSpaceIDs]
    angles = np.linspace(0, 2 * np.pi, len(archSpaceIDs), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    if not sepSamps:
        fig = plt.figure(figsize=figSize, dpi=dpi)
        ax = fig.add_subplot(111, polar=True)
        if showLegend:
            legend = []

    for i, archID in enumerate(archIDs):
        if sepSamps:
            fig = plt.figure(figsize=figSize, dpi=dpi)
            ax = fig.add_subplot(111, polar=True)

        c = sns.color_palette("husl", len(archIDs))[i]
        xClose = AA.closeMatch[archID + 1][1][archSpaceIDs]
        xClose = np.concatenate((xClose, [xClose[0]]))

        ax.plot(angles, xClose, ".-", linewidth=1, markersize=5, zorder=-1, color=c)
        ax.fill(angles, xClose, alpha=0.2, zorder=-2, color=c)
        ax.set_rticks([0.2, 0.4, 0.6, 0.8])
        ax.set_rlim(0.0, 1.0)

        if showLabel:
            if not labelAll and len(archSpaceIDs) < 8:
                labelAll = True
            labelIdxs = (
                list(range(len(angles) - 1))
                if labelAll
                else np.round(np.linspace(0, len(angles) - 1, 9)[:-1]).astype(int)
            )
            ax.set_thetagrids(angles[labelIdxs] * 180.0 / np.pi, [labels[idx] for idx in labelIdxs])
        else:
            ax.set_thetagrids(angles * 180.0 / np.pi, [""] * len(angles))

        if title:
            ax.set_title(title)
        ax.grid(linestyle="dotted", color="k", alpha=0.3, linewidth=1)
        ax.set_facecolor("#EAECEE")

        if not sepSamps:
            if showLegend:
                legend.extend([f"A{archID + 1}", "_"])
        else:
            _ensure_figs_dir()
            plt.savefig(
                f"{FIGS_DIR_PATH}/{figNamePrefix}_closeMatch_A{archID + 1}.png", bbox_inches="tight"
            )

    if not sepSamps:
        if showLegend:
            ax.legend(legend, loc="center", bbox_to_anchor=(1.2, 0.5), ncol=1)
        _ensure_figs_dir()
        plt.savefig(f"{FIGS_DIR_PATH}/{figNamePrefix}_closeMatch.png", bbox_inches="tight")


def plotTSNE(
    X: np.ndarray,
    figNamePrefix: str = "",
    figSize: tuple[int, int] = (3, 3),
    numComponents: int = 2,
    markIdxs: list[int] = [],
    markerSize: float = 1.0,
    colourInstances: bool = False,
    perplexity: float = 30.0,
    earlyExaggeration: float = 12.0,
    learningRate: str | float = "auto",
    nIter: int = 1000,
    angle: float = 0.5,
    metric: str = "euclidean",
    init: str = "pca",
    method: str = "barnes_hut",
    minGradNorm: float = 1e-7,
    nIterWithoutProgress: int = 300,
    nJobs: int = NUM_JOBS,
    randomState: int = RANDOM_STATE,
):
    """Conducts t-SNE and visualizes the results.

    Args:
        X: The dataset.
        figNamePrefix: Prefix for the saved figure filename.
        figSize: Figure size.
        numComponents: Number of components for t-SNE.
        markIdxs: Indices of samples to highlight.
        markerSize: Size of markers.
        colourInstances: Whether to color points by their index.
        perplexity: t-SNE perplexity.
        earlyExaggeration: t-SNE early exaggeration.
        learningRate: t-SNE learning rate.
        nIter: Maximum number of iterations.
        angle: t-SNE angle (Barnes-Hut).
        metric: Distance metric.
        init: Initialization method.
        method: t-SNE method.
        minGradNorm: Minimum gradient norm for stopping.
        nIterWithoutProgress: Iterations without progress for stopping.
        nJobs: Number of parallel jobs.
        randomState: Random seed.
    """
    n_samples = X.shape[0]
    if perplexity >= n_samples:
        perplexity = max(1.0, n_samples - 1.0)

    reducer = TSNE(
        n_components=numComponents,
        init=init,
        method=method,
        angle=angle,
        random_state=randomState,
        perplexity=perplexity,
        early_exaggeration=earlyExaggeration,
        learning_rate=learningRate,
        max_iter=nIter,
        n_iter_without_progress=nIterWithoutProgress,
        min_grad_norm=minGradNorm,
        metric=metric,
        verbose=0,
        n_jobs=nJobs,
    )
    embedding = reducer.fit_transform(X)

    plt.figure(figsize=figSize, dpi=DPI)
    if colourInstances:
        c, palette = range(len(X)), PALETTE
    else:
        c, palette = "k", None

    plt.scatter(x=embedding[:, 0], y=embedding[:, 1], s=markerSize, c=c, cmap=palette)
    if markIdxs:
        plt.scatter(
            x=embedding[markIdxs, 0],
            y=embedding[markIdxs, 1],
            marker="D",
            s=markerSize * 30,
            facecolor="r",
            edgecolor="k",
            linewidth=0.8,
        )

    plt.grid(linestyle="dotted")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.gca().set_aspect("equal", "datalim")
    _ensure_figs_dir()
    plt.savefig(f"{FIGS_DIR_PATH}/{figNamePrefix}_tSNE.png", bbox_inches="tight")
