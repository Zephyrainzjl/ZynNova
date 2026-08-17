from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array, finite_columns, finite_xy, make_labels


@register_plot(category="biology", aliases=("volcano",))
def volcano_plot(
    log2_fold_change: Any,
    p_value: Any,
    *,
    labels: Sequence[str] | None = None,
    fold_change_threshold: float = 1.0,
    p_value_threshold: float = 0.05,
    adjusted_p: bool = False,
    annotate_top: int = 10,
    highlight: Any | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Differential-expression volcano plot with highlighting and q-value labels."""
    fold,p,mask=finite_xy(log2_fold_change,p_value); p=np.clip(p,np.finfo(float).tiny,1.0); y=-np.log10(p)
    significant=(np.abs(fold)>=fold_change_threshold)&(p<=p_value_threshold); up=significant&(fold>0); down=significant&(fold<0)
    highlight_mask=np.zeros(fold.size,dtype=bool)
    if highlight is not None:
        raw=np.asarray(highlight)
        if raw.dtype==bool:
            if raw.size!=mask.size: raise ValueError("boolean highlight must match original inputs")
            highlight_mask=raw.reshape(-1)[mask]
        elif labels is not None:
            label_arr=np.asarray(labels).reshape(-1)[mask]; highlight_mask=np.isin(label_arr,raw.reshape(-1))
        else: raise ValueError("non-boolean highlight requires labels")
    ylabel=r"$-\log_{10}(q)$" if adjusted_p else r"$-\log_{10}(p)$"
    cfg=coerce_config(config,xlabel=r"$\log_2$ fold change",ylabel=ylabel)
    with theme_context(theme):
        fig,axis,resolved=create_axes(ax=ax,config=cfg,theme=theme)
        nonsig=axis.scatter(fold[~significant&~highlight_mask],y[~significant&~highlight_mask],s=9,color="0.7",alpha=0.5,rasterized=cfg.rasterized,label="Not significant")
        up_artist=axis.scatter(fold[up&~highlight_mask],y[up&~highlight_mask],s=11,color=resolved.colors[1],alpha=0.7,rasterized=cfg.rasterized,label="Up")
        down_artist=axis.scatter(fold[down&~highlight_mask],y[down&~highlight_mask],s=11,color=resolved.colors[0],alpha=0.7,rasterized=cfg.rasterized,label="Down")
        highlight_artist=axis.scatter(fold[highlight_mask],y[highlight_mask],s=30,facecolors="none",edgecolors="black",linewidths=0.9,label="Highlighted") if highlight_mask.any() else None
        thresholds=[axis.axvline(fold_change_threshold,color="black",linestyle="--",linewidth=0.7),axis.axvline(-fold_change_threshold,color="black",linestyle="--",linewidth=0.7),axis.axhline(-np.log10(p_value_threshold),color="black",linestyle="--",linewidth=0.7)]
        annotations=[]
        if labels is not None and annotate_top>0:
            label_arr=np.asarray(labels).reshape(-1)[mask]; score=y+np.abs(fold); candidates=np.flatnonzero(significant|highlight_mask); selected=candidates[np.argsort(score[candidates])[-annotate_top:]] if candidates.size else []
            for index in selected: annotations.append(axis.annotate(str(label_arr[index]),(fold[index],y[index]),xytext=(3,3),textcoords="offset points",fontsize="x-small"))
        return finalize(fig,axis,config=cfg,artists={"nonsignificant":nonsig,"up":up_artist,"down":down_artist,"highlight":highlight_artist,"thresholds":thresholds,"annotations":annotations},data={"log2_fold_change":fold,"p_value":p,"adjusted_p":adjusted_p,"significant":significant,"highlight":highlight_mask},metrics={"significant_count":int(significant.sum()),"up_count":int(up.sum()),"down_count":int(down.sum()),"highlight_count":int(highlight_mask.sum())},theme=theme)


@register_plot(category="biology", aliases=("ma",))
def ma_plot(
    mean_expression: Any,
    log2_fold_change: Any,
    *,
    significant: Any | None = None,
    labels: Sequence[str] | None = None,
    annotate_top: int = 8,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """MA plot for abundance-dependent differential effects."""
    mean, fold, mask = finite_xy(mean_expression, log2_fold_change)
    sig = np.zeros(mean.size, dtype=bool) if significant is None else np.asarray(significant, dtype=bool)[mask]
    cfg = coerce_config(config, xlabel="Mean expression", ylabel=r"$\log_2$ fold change", xscale="log")
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        nonsig = axis.scatter(mean[~sig], fold[~sig], s=8, color="0.7", alpha=0.45, rasterized=True)
        sig_artist = axis.scatter(mean[sig], fold[sig], s=10, color=resolved.colors[1], alpha=0.7, rasterized=True, label="Significant")
        axis.axhline(0.0, color="black", linewidth=0.7)
        if labels is not None and annotate_top > 0 and sig.any():
            label_arr = np.asarray(labels)[mask]
            selected = np.flatnonzero(sig)[np.argsort(np.abs(fold[sig]))[-annotate_top:]]
            for index in selected:
                axis.annotate(str(label_arr[index]), (mean[index], fold[index]), xytext=(3, 3), textcoords="offset points", fontsize="x-small")
        return finalize(fig, axis, config=cfg, artists={"nonsignificant": nonsig, "significant": sig_artist}, data={"mean_expression": mean, "log2_fold_change": fold, "significant": sig})


@register_plot(category="biology", aliases=("expression-heatmap", "clustered-heatmap"))
def expression_heatmap(
    expression: Any,
    *,
    row_labels: Sequence[str] | None = None,
    column_labels: Sequence[str] | None = None,
    row_cluster: bool = True,
    column_cluster: bool = True,
    z_score: str | None = "row",
    cmap: str = "coolwarm",
    vlim: float | None = 3.0,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Clustered gene/protein/feature expression heatmap without seaborn."""
    matrix = np.asarray(expression, dtype=float)
    rows = np.asarray(make_labels(row_labels, matrix.shape[0], "Feature"), dtype=object)
    cols = np.asarray(make_labels(column_labels, matrix.shape[1], "Sample"), dtype=object)
    if z_score == "row":
        matrix = (matrix - np.nanmean(matrix, axis=1, keepdims=True)) / np.maximum(np.nanstd(matrix, axis=1, keepdims=True), 1e-15)
    elif z_score == "column":
        matrix = (matrix - np.nanmean(matrix, axis=0, keepdims=True)) / np.maximum(np.nanstd(matrix, axis=0, keepdims=True), 1e-15)
    row_order = np.arange(matrix.shape[0])
    column_order = np.arange(matrix.shape[1])
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage

        if row_cluster and matrix.shape[0] > 2:
            row_order = leaves_list(linkage(np.nan_to_num(matrix), method="average", metric="euclidean"))
        if column_cluster and matrix.shape[1] > 2:
            column_order = leaves_list(linkage(np.nan_to_num(matrix.T), method="average", metric="euclidean"))
    except Exception:
        pass
    matrix = matrix[np.ix_(row_order, column_order)]
    rows, cols = rows[row_order], cols[column_order]
    cfg = coerce_config(config, figsize=(max(5.0, matrix.shape[1] * 0.18), max(4.0, matrix.shape[0] * 0.16)))
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.imshow(matrix, cmap=cmap, aspect="auto", vmin=-vlim if vlim else None, vmax=vlim)
        if len(cols) <= 80:
            axis.set_xticks(np.arange(len(cols)), cols, rotation=90)
        if len(rows) <= 100:
            axis.set_yticks(np.arange(len(rows)), rows)
        colorbar = fig.colorbar(image, ax=axis, label="Scaled expression" if z_score else "Expression")
        return finalize(fig, axis, config=cfg, artists={"image": image, "colorbar": colorbar}, data={"expression": matrix, "row_labels": rows, "column_labels": cols, "row_order": row_order, "column_order": column_order})


@register_plot(category="biology", aliases=("single-cell-dotplot", "marker-dotplot"))
def expression_dot_plot(
    mean_expression: Any,
    fraction_expressing: Any,
    *,
    feature_names: Sequence[str],
    group_names: Sequence[str],
    size_range: tuple[float, float] = (5.0, 180.0),
    cmap: str = "viridis",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Single-cell marker dot plot: color=mean, size=fraction expressing."""
    mean = np.asarray(mean_expression, dtype=float)
    fraction = np.asarray(fraction_expressing, dtype=float)
    if mean.shape != fraction.shape or mean.shape != (len(group_names), len(feature_names)):
        raise ValueError("matrices must have shape (groups, features)")
    xx, yy = np.meshgrid(np.arange(len(feature_names)), np.arange(len(group_names)))
    minimum, maximum = size_range
    sizes = minimum + (maximum - minimum) * np.clip(fraction, 0, 1)
    cfg = coerce_config(config, figsize=(max(5.0, len(feature_names) * 0.35), max(3.0, len(group_names) * 0.35)))
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        points = axis.scatter(xx.ravel(), yy.ravel(), c=mean.ravel(), s=sizes.ravel(), cmap=cmap, edgecolors="0.4", linewidths=0.2)
        axis.set_xticks(np.arange(len(feature_names)), feature_names, rotation=90)
        axis.set_yticks(np.arange(len(group_names)), group_names)
        axis.invert_yaxis()
        colorbar = fig.colorbar(points, ax=axis, label="Mean expression")
        return finalize(fig, axis, config=cfg, artists={"points": points, "colorbar": colorbar}, data={"mean_expression": mean, "fraction_expressing": fraction})


@register_plot(category="biology", aliases=("spatial-expression", "spatial-omics"))
def spatial_expression_plot(
    coordinates: Any,
    values: Any,
    *,
    image: Any | None = None,
    spot_size: Any | float = 10.0,
    cmap: str = "viridis",
    alpha: float = 0.85,
    invert_y: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Spatial transcriptomic/proteomic values over tissue or microscopy image."""
    xy = np.asarray(coordinates, dtype=float)
    scalar = as_array(values)
    cfg = coerce_config(config, equal_aspect=True)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image_artist = axis.imshow(image, origin="upper") if image is not None else None
        points = axis.scatter(xy[:, 0], xy[:, 1], c=scalar, s=spot_size, cmap=cmap, alpha=alpha, edgecolors="none", rasterized=cfg.rasterized)
        if invert_y and image is None:
            axis.invert_yaxis()
        axis.set_axis_off()
        colorbar = fig.colorbar(points, ax=axis, label="Expression")
        return finalize(fig, axis, config=cfg, artists={"image": image_artist, "points": points, "colorbar": colorbar}, data={"coordinates": xy, "values": scalar})


@register_plot(category="biology", aliases=("kaplan-meier", "survival"))
def survival_curve_plot(
    time: Any,
    event: Any,
    *,
    groups: Any | None = None,
    confidence: bool = True,
    show_censors: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Kaplan–Meier survival curves with Greenwood confidence bands."""
    t = as_array(time)
    e = as_array(event, dtype=int)
    group_arr = np.zeros(t.size, dtype=int) if groups is None else np.asarray(groups).reshape(-1)
    cfg = coerce_config(config, xlabel="Time", ylabel="Survival probability", ylim=(0, 1.02))
    curves = {}
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        artists, bands, censors = [], [], []
        for group in dict.fromkeys(group_arr.tolist()):
            selected = group_arr == group
            tg, eg = t[selected], e[selected]
            order = np.argsort(tg)
            tg, eg = tg[order], eg[order]
            event_times = np.unique(tg[eg == 1])
            survival = 1.0
            times = [0.0]
            probabilities = [1.0]
            variance_sum = 0.0
            lower, upper = [1.0], [1.0]
            for event_time in event_times:
                at_risk = np.sum(tg >= event_time)
                events = np.sum((tg == event_time) & (eg == 1))
                survival *= 1.0 - events / max(at_risk, 1)
                if at_risk > events and events > 0:
                    variance_sum += events / (at_risk * (at_risk - events))
                se = survival * np.sqrt(max(variance_sum, 0.0))
                times.append(float(event_time))
                probabilities.append(float(survival))
                lower.append(max(0.0, survival - 1.96 * se))
                upper.append(min(1.0, survival + 1.96 * se))
            line = axis.step(times, probabilities, where="post", label=str(group))[0]
            artists.append(line)
            if confidence:
                bands.append(axis.fill_between(times, lower, upper, step="post", color=line.get_color(), alpha=0.15))
            if show_censors:
                for censor_time in tg[eg == 0]:
                    index = np.searchsorted(times, censor_time, side="right") - 1
                    censors.append(axis.plot(censor_time, probabilities[max(index, 0)], marker="+", color=line.get_color(), linestyle="none")[0])
            curves[str(group)] = {"time": np.asarray(times), "survival": np.asarray(probabilities)}
        return finalize(fig, axis, config=cfg, artists={"curves": artists, "bands": bands, "censors": censors}, data={"curves": curves})


@register_plot(category="biology", aliases=("enrichment-bubble", "pathway-enrichment"))
def enrichment_bubble_plot(
    terms: Sequence[str],
    enrichment: Any,
    p_value: Any,
    *,
    count: Any | None = None,
    top_k: int = 20,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Pathway/ontology enrichment bubble chart."""
    enrich = as_array(enrichment)
    p = np.clip(as_array(p_value), np.finfo(float).tiny, 1.0)
    term_arr = np.asarray(terms, dtype=object)
    sizes = np.full(enrich.size, 45.0) if count is None else 15.0 + 120.0 * as_array(count) / max(np.max(as_array(count)), 1e-15)
    order = np.argsort(-np.log10(p))[-min(top_k, enrich.size):]
    order = order[np.argsort(enrich[order])]
    cfg = coerce_config(config, xlabel="Enrichment score")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        points = axis.scatter(enrich[order], np.arange(order.size), c=-np.log10(p[order]), s=sizes[order], cmap="viridis", edgecolors="0.4", linewidths=0.3)
        axis.set_yticks(np.arange(order.size), term_arr[order])
        colorbar = fig.colorbar(points, ax=axis, label=r"$-\log_{10}(p)$")
        return finalize(fig, axis, config=cfg, artists={"points": points, "colorbar": colorbar}, data={"terms": term_arr[order], "enrichment": enrich[order], "p_value": p[order], "sizes": sizes[order]})


@register_plot(category="biology", aliases=("manhattan",))
def manhattan_plot(
    chromosome: Any,
    position: Any,
    p_value: Any,
    *,
    labels: Sequence[str] | None = None,
    genome_wide_threshold: float = 5e-8,
    suggestive_threshold: float | None = 1e-5,
    annotate_top: int = 8,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """GWAS Manhattan plot with alternating chromosomes and hit labels."""
    chrom = np.asarray(chromosome).reshape(-1)
    pos = as_array(position)
    p = np.clip(as_array(p_value), np.finfo(float).tiny, 1.0)
    unique = list(dict.fromkeys(chrom.tolist()))
    offsets = {}
    cursor = 0.0
    x = np.empty_like(pos)
    centers = []
    for item in unique:
        selected = chrom == item
        positions = pos[selected]
        x[selected] = positions - positions.min() + cursor
        centers.append(0.5 * (x[selected].min() + x[selected].max()))
        offsets[item] = cursor
        cursor = x[selected].max() + max(np.ptp(positions) * 0.03, 1.0)
    y = -np.log10(p)
    cfg = coerce_config(config, xlabel="Chromosome", ylabel=r"$-\log_{10}(p)$")
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        scatters = []
        for index, item in enumerate(unique):
            selected = chrom == item
            scatters.append(axis.scatter(x[selected], y[selected], s=6, color=resolved.colors[index % 2], alpha=0.7, rasterized=True))
        axis.axhline(-np.log10(genome_wide_threshold), color=resolved.colors[1], linestyle="--", linewidth=0.9, label="Genome-wide")
        if suggestive_threshold:
            axis.axhline(-np.log10(suggestive_threshold), color="0.4", linestyle=":", linewidth=0.8, label="Suggestive")
        axis.set_xticks(centers, [str(item) for item in unique])
        if labels is not None and annotate_top > 0:
            label_arr = np.asarray(labels)
            selected = np.argsort(y)[-annotate_top:]
            for index in selected:
                axis.annotate(str(label_arr[index]), (x[index], y[index]), xytext=(2, 3), textcoords="offset points", fontsize="x-small")
        return finalize(fig, axis, config=cfg, artists={"chromosomes": scatters}, data={"x": x, "minus_log10_p": y, "chromosome": chrom, "position": pos})
