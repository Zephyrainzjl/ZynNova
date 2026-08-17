from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array, finite_xy, make_labels


@register_plot(category="electrochemistry", aliases=("voltage-capacity", "charge-discharge"))
def voltage_capacity_plot(
    capacity: Any,
    voltage: Any,
    *,
    cycles: Any | None = None,
    direction: Any | None = None,
    normalize_capacity: bool = False,
    cmap: str = "viridis",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Charge/discharge voltage profiles colored by cycle or rate."""
    q, v, mask = finite_xy(capacity, voltage)
    if normalize_capacity:
        q = q / max(np.nanmax(q), 1e-15)
    cycle_arr = None if cycles is None else np.asarray(cycles).reshape(-1)[mask]
    direction_arr = None if direction is None else np.asarray(direction).reshape(-1)[mask]
    if cycle_arr is not None and cycle_arr.size != q.size: raise ValueError("cycles must match capacity")
    if direction_arr is not None and direction_arr.size != q.size: raise ValueError("direction must match capacity")
    cfg = coerce_config(config, xlabel="Normalized capacity" if normalize_capacity else "Capacity", ylabel="Voltage (V)")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        import matplotlib.colors as mcolors
        cmap_obj = __import__("matplotlib").colormaps[cmap]
        unique_cycles = [None] if cycle_arr is None else list(dict.fromkeys(cycle_arr.tolist()))
        numeric_cycles = cycle_arr is not None and np.issubdtype(np.asarray(cycle_arr).dtype, np.number)
        norm = mcolors.Normalize(float(np.nanmin(cycle_arr)), float(np.nanmax(cycle_arr))) if numeric_cycles and len(unique_cycles)>1 else None
        artists=[]
        style_map={"charge":"-", "discharge":"--", "chg":"-", "dchg":"--", 1:"-", -1:"--"}
        for ci, cycle in enumerate(unique_cycles):
            cmask = np.ones(q.size,dtype=bool) if cycle is None else cycle_arr==cycle
            dirs=[None] if direction_arr is None else list(dict.fromkeys(direction_arr[cmask].tolist()))
            color = cmap_obj(norm(float(cycle))) if norm is not None else cmap_obj(ci/max(len(unique_cycles)-1,1))
            for direct in dirs:
                selected=cmask if direct is None else cmask & (direction_arr==direct)
                if not np.any(selected): continue
                dkey=direct.lower() if isinstance(direct,str) else direct
                linestyle=style_map.get(dkey,"-" if len(dirs)==1 else ["-","--","-.",":"][dirs.index(direct)%4])
                label_parts=[]
                if cycle is not None: label_parts.append(f"Cycle {cycle}")
                if direct is not None: label_parts.append(str(direct))
                artists.append(axis.plot(q[selected],v[selected],color=color,linestyle=linestyle,label=" · ".join(label_parts) or None)[0])
        colorbar=None
        if norm is not None:
            from matplotlib.cm import ScalarMappable
            colorbar=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap_obj),ax=axis,label="Cycle")
        return finalize(fig, axis, config=cfg, artists={"profiles":artists,"colorbar":colorbar}, data={"capacity":q,"voltage":v,"cycles":cycle_arr,"direction":direction_arr}, theme=theme)


@register_plot(category="electrochemistry", aliases=("dqdv", "differential-capacity"))
def differential_capacity_plot(
    capacity: Any,
    voltage: Any,
    *,
    smooth_window: int = 11,
    derivative: str = "dQdV",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Differential capacity dQ/dV or differential voltage dV/dQ."""
    q, v, _ = finite_xy(capacity, voltage)
    order = np.argsort(v if derivative.lower() == "dqdv" else q)
    q, v = q[order], v[order]
    if smooth_window > 2:
        try:
            from scipy.signal import savgol_filter

            window = min(smooth_window if smooth_window % 2 else smooth_window + 1, len(q) - (1 - len(q) % 2))
            q_s = savgol_filter(q, max(window, 3), 2) if window >= 3 else q
            v_s = savgol_filter(v, max(window, 3), 2) if window >= 3 else v
        except Exception:
            q_s, v_s = q, v
    else:
        q_s, v_s = q, v
    if derivative.lower() == "dvdq":
        x, y, ylabel = q_s, np.gradient(v_s, q_s), "dV/dQ"
    else:
        x, y, ylabel = v_s, np.gradient(q_s, v_s), "dQ/dV"
    cfg = coerce_config(config, xlabel="Capacity" if derivative.lower() == "dvdq" else "Voltage (V)", ylabel=ylabel)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        line = axis.plot(x, y)[0]
        axis.axhline(0.0, color="black", linewidth=0.6)
        return finalize(fig, axis, config=cfg, artists={"curve": line}, data={"x": x, "derivative": y, "capacity": q_s, "voltage": v_s})


@register_plot(category="electrochemistry", aliases=("nyquist", "eis-nyquist"))
def nyquist_plot(
    real: Any,
    imaginary: Any,
    *,
    frequency: Any | None = None,
    labels: Sequence[str] | None = None,
    negate_imaginary: bool = True,
    annotate_frequencies: Sequence[float] | None = None,
    equal_aspect: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Electrochemical impedance Nyquist plot with frequency annotations."""
    re = np.asarray(real, dtype=float)
    im = np.asarray(imaginary, dtype=float)
    if re.ndim == 1:
        re, im = re[None, :], im[None, :]
    names = make_labels(labels, re.shape[0], "Spectrum")
    cfg = coerce_config(config, xlabel=r"$Z'$", ylabel=r"$-Z''$" if negate_imaginary else r"$Z''$", equal_aspect=equal_aspect)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines = []
        for row_re, row_im, name in zip(re, im, names):
            y = -row_im if negate_imaginary else row_im
            line = axis.plot(row_re, y, marker="o", markersize=3, label=name)[0]
            lines.append(line)
        if frequency is not None and annotate_frequencies:
            freq = as_array(frequency)
            for target in annotate_frequencies:
                index = int(np.argmin(np.abs(freq - target)))
                axis.annotate(f"{freq[index]:.2g} Hz", (re[0, index], (-im[0, index] if negate_imaginary else im[0, index])), xytext=(4, 4), textcoords="offset points", fontsize="x-small")
        return finalize(fig, axis, config=cfg, artists={"spectra": lines}, data={"real": re, "imaginary": im, "frequency": frequency})


@register_plot(category="electrochemistry", aliases=("bode", "eis-bode"))
def bode_plot(
    frequency: Any,
    real: Any,
    imaginary: Any,
    *,
    labels: Sequence[str] | None = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Two-panel Bode magnitude and phase plot."""
    freq = as_array(frequency)
    re = np.asarray(real, dtype=float)
    im = np.asarray(imaginary, dtype=float)
    if re.ndim == 1:
        re, im = re[None, :], im[None, :]
    names = make_labels(labels, re.shape[0], "Spectrum")
    magnitude = np.sqrt(re**2 + im**2)
    phase = np.rad2deg(np.arctan2(im, re))
    cfg = coerce_config(config, figsize=(6.5, 5.0))
    with theme_context(theme):
        fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=2, ncols=1, sharex=True)
        lines_mag, lines_phase = [], []
        for mag, ph, name in zip(magnitude, phase, names):
            line = axes[0].plot(freq, mag, label=name)[0]
            lines_mag.append(line)
            lines_phase.append(axes[1].plot(freq, ph, color=line.get_color(), label=name)[0])
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("|Z|")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Phase (deg)")
        return finalize(fig, axes, config=cfg, artists={"magnitude": lines_mag, "phase": lines_phase}, data={"frequency": freq, "magnitude": magnitude, "phase": phase})


@register_plot(category="electrochemistry", aliases=("cyclic-voltammetry", "cv"))
def cyclic_voltammetry_plot(
    potential: Any,
    current: Any,
    *,
    cycles: Any | None = None,
    scan_rate: Any | None = None,
    normalize_current: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Cyclic voltammogram with cycle- and scan-rate-separated traces."""
    potential_arr, current_arr, mask = finite_xy(potential,current)
    if normalize_current: current_arr=current_arr/max(np.nanmax(np.abs(current_arr)),1e-15)
    cycle_arr=None if cycles is None else np.asarray(cycles).reshape(-1)[mask]
    rate_arr=None
    if scan_rate is not None:
        raw=np.asarray(scan_rate)
        if raw.ndim==0: rate_arr=np.full(potential_arr.size,raw.item())
        elif raw.size==potential_arr.size: rate_arr=raw.reshape(-1)[mask] if raw.size==mask.size else raw.reshape(-1)
        elif cycle_arr is not None and raw.size==len(dict.fromkeys(cycle_arr.tolist())):
            mapping=dict(zip(dict.fromkeys(cycle_arr.tolist()),raw.reshape(-1))); rate_arr=np.asarray([mapping[c] for c in cycle_arr])
        else: raise ValueError("scan_rate must be scalar, per point, or per cycle")
    cfg=coerce_config(config,xlabel="Potential (V)",ylabel="Normalized current" if normalize_current else "Current")
    with theme_context(theme):
        fig,axis,resolved=create_axes(ax=ax,config=cfg,theme=theme)
        lines=[]; unique=[None] if cycle_arr is None else list(dict.fromkeys(cycle_arr.tolist()))
        for i,cycle in enumerate(unique):
            selected=np.ones(potential_arr.size,dtype=bool) if cycle is None else cycle_arr==cycle
            rate=None if rate_arr is None else rate_arr[selected][0]
            label=[]
            if cycle is not None: label.append(f"Cycle {cycle}")
            if rate is not None: label.append(f"{rate:g} mV s$^{{-1}}$" if np.issubdtype(np.asarray(rate).dtype,np.number) else str(rate))
            lines.append(axis.plot(potential_arr[selected],current_arr[selected],label=" · ".join(label) or None,color=resolved.colors[i%len(resolved.colors)])[0])
        zero=axis.axhline(0,color="black",linewidth=0.6)
        return finalize(fig,axis,config=cfg,artists={"cycles":lines,"zero":zero},data={"potential":potential_arr,"current":current_arr,"cycles":cycle_arr,"scan_rate":rate_arr},theme=theme)


@register_plot(category="electrochemistry", aliases=("ragone",))
def ragone_plot(
    energy_density: Any,
    power_density: Any,
    *,
    labels: Any | None = None,
    color: Any | None = None,
    size: Any | float = 30.0,
    annotate: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Ragone energy–power trade-off map."""
    energy, power, mask = finite_xy(energy_density, power_density)
    label_arr = None if labels is None else np.asarray(labels)[mask]
    cfg = coerce_config(config, xlabel="Specific energy", ylabel="Specific power", xscale="log", yscale="log")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        points = axis.scatter(energy, power, c=color if color is not None else "0.55", s=size, cmap="viridis" if color is not None else None, alpha=0.75, edgecolors="none")
        colorbar = fig.colorbar(points, ax=axis, label="Color value") if color is not None and np.issubdtype(np.asarray(color).dtype, np.number) else None
        if annotate and label_arr is not None:
            for x, y, label in zip(energy, power, label_arr):
                axis.annotate(str(label), (x, y), xytext=(3, 3), textcoords="offset points")
        return finalize(fig, axis, config=cfg, artists={"points": points, "colorbar": colorbar}, data={"energy_density": energy, "power_density": power})


@register_plot(category="electrochemistry", aliases=("cycling-retention",))
def cycling_performance_plot(
    cycle: Any,
    capacity: Any,
    *,
    coulombic_efficiency: Any | None = None,
    groups: Any | None = None,
    initial_normalization: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Capacity retention and coulombic efficiency versus cycle number."""
    x, cap, mask=finite_xy(cycle,capacity)
    group_arr=None if groups is None else np.asarray(groups).reshape(-1)[mask]
    ce=None if coulombic_efficiency is None else np.asarray(coulombic_efficiency,dtype=float).reshape(-1)[mask]
    cfg=coerce_config(config,xlabel="Cycle",ylabel="Capacity retention (%)" if initial_normalization else "Capacity")
    with theme_context(theme):
        fig,axis,resolved=create_axes(ax=ax,config=cfg,theme=theme)
        capacity_lines=[]; efficiency_lines=[]
        unique=[None] if group_arr is None else list(dict.fromkeys(group_arr.tolist()))
        efficiency_axis=axis.twinx() if ce is not None else None
        normalized_cap=cap.copy()
        for i,group in enumerate(unique):
            selected=np.ones(x.size,dtype=bool) if group is None else group_arr==group
            ordered=np.flatnonzero(selected)[np.argsort(x[selected])]
            curve=cap[ordered].copy()
            if initial_normalization: curve=curve/max(curve[0],1e-15)*100; normalized_cap[ordered]=curve
            label="Capacity" if group is None else str(group)
            line=axis.plot(x[ordered],curve,marker="o",markersize=2.5,label=label,color=resolved.colors[i%len(resolved.colors)])[0]
            capacity_lines.append(line)
            if ce is not None:
                efficiency_lines.append(efficiency_axis.plot(x[ordered],ce[ordered],linestyle="--",color=line.get_color(),alpha=0.6,label=f"{label} CE")[0])
        if efficiency_axis is not None: efficiency_axis.set_ylabel("Coulombic efficiency (%)")
        final_retention=float(normalized_cap[-1]) if initial_normalization else float(cap[-1]/max(cap[0],1e-15)*100)
        return finalize(fig,np.asarray([axis,efficiency_axis],dtype=object) if efficiency_axis is not None else axis,config=cfg,artists={"capacity":capacity_lines,"efficiency":efficiency_lines},data={"cycle":x,"capacity":normalized_cap,"raw_capacity":cap,"groups":group_arr,"coulombic_efficiency":ce},metrics={"final_retention":final_retention},theme=theme)


@register_plot(category="electrochemistry", aliases=("rate-capability",))
def rate_capability_plot(
    rate: Any,
    capacity: Any,
    *,
    groups: Any | None = None,
    connect: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Rate capability across C-rates or current densities."""
    rate_arr=np.asarray(rate).reshape(-1); capacity_arr=as_array(capacity)
    if rate_arr.size!=capacity_arr.size: raise ValueError("rate and capacity must match")
    group_arr=None if groups is None else np.asarray(groups).reshape(-1)
    if group_arr is not None and group_arr.size!=rate_arr.size: raise ValueError("groups must match rate")
    categories=list(dict.fromkeys(rate_arr.tolist())); positions={value:i for i,value in enumerate(categories)}
    cfg=coerce_config(config,xlabel="Rate",ylabel="Capacity")
    with theme_context(theme):
        fig,axis,resolved=create_axes(ax=ax,config=cfg,theme=theme); artists=[]
        unique=[None] if group_arr is None else list(dict.fromkeys(group_arr.tolist()))
        for i,group in enumerate(unique):
            selected=np.ones(rate_arr.size,dtype=bool) if group is None else group_arr==group
            x=np.asarray([positions[v] for v in rate_arr[selected]],dtype=float)
            label=None if group is None else str(group)
            if connect: artists.append(axis.plot(x,capacity_arr[selected],marker="o",label=label,color=resolved.colors[i%len(resolved.colors)])[0])
            else:
                width=0.8/max(len(unique),1); offset=(i-(len(unique)-1)/2)*width
                artists.append(axis.bar(x+offset,capacity_arr[selected],width=width,label=label,color=resolved.colors[i%len(resolved.colors)]))
        axis.set_xticks(np.arange(len(categories)),[str(item) for item in categories])
        return finalize(fig,axis,config=cfg,artists={"values":artists},data={"rate":rate_arr,"capacity":capacity_arr,"groups":group_arr},theme=theme)


@register_plot(category="electrochemistry", aliases=("soc-temperature-map", "battery-operating-map"))
def soc_temperature_map(
    soc: Any,
    temperature: Any,
    value: Any,
    *,
    levels: int = 30,
    cmap: str = "viridis",
    constraints: Sequence[tuple[Any, str]] | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """SOC–temperature performance, degradation or risk surface."""
    soc_arr=np.asarray(soc,dtype=float); temp_arr=np.asarray(temperature,dtype=float); values=np.asarray(value,dtype=float)
    xx,yy=np.meshgrid(soc_arr,temp_arr) if soc_arr.ndim==temp_arr.ndim==1 else (soc_arr,temp_arr)
    if values.shape!=xx.shape: raise ValueError("value shape must match SOC-temperature grid")
    cfg=coerce_config(config,xlabel="SOC",ylabel="Temperature (K)")
    with theme_context(theme):
        fig,axis,resolved=create_axes(ax=ax,config=cfg,theme=theme)
        surface=axis.contourf(xx,yy,values,levels=levels,cmap=cmap)
        contours=axis.contour(xx,yy,values,levels=max(5,levels//5),colors="black",linewidths=0.4,alpha=0.5)
        constraint_artists=[]
        if constraints:
            for i,(constraint,label) in enumerate(constraints):
                field=np.asarray(constraint,dtype=float)
                if field.shape!=xx.shape: raise ValueError(f"constraint {label!r} shape must match value")
                finite=field[np.isfinite(field)]; level=0.5 if finite.size and finite.min()>=0 and finite.max()<=1 else 0.0
                constraint_color=resolved.colors[(i+1)%len(resolved.colors)]
                cs=axis.contour(xx,yy,field,levels=[level],colors=[constraint_color],linewidths=1.6)
                proxy=axis.plot([],[],color=constraint_color,linewidth=1.6,label=str(label))[0]
                constraint_artists.append({"contour":cs,"legend_proxy":proxy})
        colorbar=fig.colorbar(surface,ax=axis,label="Value")
        return finalize(fig,axis,config=cfg,artists={"surface":surface,"contours":contours,"constraints":constraint_artists,"colorbar":colorbar},data={"soc":xx,"temperature":yy,"value":values,"constraints":constraints},theme=theme)
