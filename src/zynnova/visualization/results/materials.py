from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array, finite_xy, make_labels


@register_plot(category="materials", aliases=("phase-map", "phase-diagram"))
def phase_diagram_2d(
    x: Any,
    y: Any,
    phase: Any,
    *,
    phase_names: Mapping[Any, str] | None = None,
    boundaries: bool = True,
    points: bool = False,
    cmap: str = "tab20",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Categorical 2-D phase diagram from grids or scattered samples."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    phase_arr = np.asarray(phase)
    if x_arr.ndim == y_arr.ndim == 1 and phase_arr.shape == (y_arr.size, x_arr.size):
        xx, yy = np.meshgrid(x_arr, y_arr)
        grid = phase_arr
    elif phase_arr.ndim == 1:
        try:
            from scipy.interpolate import griddata

            xi = np.linspace(np.nanmin(x_arr), np.nanmax(x_arr), 250)
            yi = np.linspace(np.nanmin(y_arr), np.nanmax(y_arr), 250)
            xx, yy = np.meshgrid(xi, yi)
            unique = np.unique(phase_arr)
            encoded = np.searchsorted(unique, phase_arr)
            grid = griddata((as_array(x_arr), as_array(y_arr)), encoded, (xx, yy), method="nearest")
        except Exception as exc:
            raise ValueError("scattered phase diagrams require scipy") from exc
    else:
        xx, yy, grid = x_arr, y_arr, phase_arr
    unique_grid = np.unique(grid[np.isfinite(grid)])
    cfg = coerce_config(config)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        mesh = axis.pcolormesh(xx, yy, grid, cmap=cmap, shading="auto")
        contour = axis.contour(xx, yy, grid, levels=np.arange(np.nanmin(grid), np.nanmax(grid) + 1) + 0.5, colors="black", linewidths=0.6) if boundaries and unique_grid.size > 1 else None
        scatter = axis.scatter(as_array(x_arr), as_array(y_arr), c=as_array(phase_arr), cmap=cmap, s=8, edgecolors="none") if points and phase_arr.ndim == 1 else None
        colorbar = fig.colorbar(mesh, ax=axis, ticks=unique_grid)
        if phase_names:
            colorbar.ax.set_yticklabels([phase_names.get(value, str(value)) for value in unique_grid])
        colorbar.set_label("Phase")
        return finalize(fig, axis, config=cfg, artists={"mesh": mesh, "boundaries": contour, "points": scatter, "colorbar": colorbar}, data={"x": xx, "y": yy, "phase": grid})


@register_plot(category="materials", aliases=("convex-hull", "formation-energy-hull"))
def convex_hull_plot(
    composition: Any,
    formation_energy: Any,
    *,
    labels: Sequence[str] | None = None,
    stable_mask: Any | None = None,
    annotate_stable: bool = True,
    energy_above_hull: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Binary composition–formation-energy convex hull with stability labels."""
    x, energy, mask = finite_xy(composition, formation_energy)
    order = np.argsort(x)
    x, energy = x[order], energy[order]
    if stable_mask is None:
        hull_indices = [0]
        for index in range(1, len(x)):
            while len(hull_indices) >= 2:
                i, j = hull_indices[-2], hull_indices[-1]
                slope_old = (energy[j] - energy[i]) / max(x[j] - x[i], 1e-15)
                slope_new = (energy[index] - energy[j]) / max(x[index] - x[j], 1e-15)
                if slope_new <= slope_old:
                    hull_indices.pop()
                else:
                    break
            hull_indices.append(index)
        stable = np.zeros(len(x), dtype=bool)
        stable[hull_indices] = True
    else:
        stable = np.asarray(stable_mask, dtype=bool)[mask][order]
        hull_indices = np.flatnonzero(stable).tolist()
    hull_x = x[stable]
    hull_energy = energy[stable]
    interpolated = np.interp(x, hull_x, hull_energy)
    above = energy - interpolated
    cfg = coerce_config(config, xlabel="Composition", ylabel="Formation energy")
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        unstable_points = axis.scatter(x[~stable], energy[~stable], s=18, alpha=0.45, label="Metastable")
        stable_points = axis.scatter(x[stable], energy[stable], s=38, color=resolved.colors[1], label="Stable")
        hull_line = axis.plot(hull_x, hull_energy, color=resolved.colors[1], linewidth=1.4, label="Convex hull")[0]
        if labels is not None and annotate_stable:
            label_arr = np.asarray(labels)[mask][order]
            for xx, yy, text in zip(x[stable], energy[stable], label_arr[stable]):
                axis.annotate(str(text), (xx, yy), xytext=(3, 4), textcoords="offset points")
        if energy_above_hull:
            for xx, yy, hull_yy in zip(x[~stable], energy[~stable], interpolated[~stable]):
                axis.plot([xx, xx], [hull_yy, yy], color="0.75", linewidth=0.5)
        return finalize(fig, axis, config=cfg, artists={"unstable": unstable_points, "stable": stable_points, "hull": hull_line}, data={"composition": x, "formation_energy": energy, "stable_mask": stable, "energy_above_hull": above}, metrics={"stable_count": int(stable.sum())})


@register_plot(category="materials", aliases=("bands",))
def band_structure_plot(
    k_distance: Any,
    energies: Any,
    *,
    fermi_level: float = 0.0,
    occupations: Any | None = None,
    spin: Any | None = None,
    high_symmetry_positions: Sequence[float] | None = None,
    high_symmetry_labels: Sequence[str] | None = None,
    energy_window: tuple[float, float] | None = None,
    linewidth: float = 0.8,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Electronic band structure with symmetry path, spin and occupations."""
    k=as_array(k_distance); band=np.asarray(energies,dtype=float)
    if band.ndim==1: band=band[:,None]
    if band.shape[0]!=k.size and band.shape[1]==k.size: band=band.T
    if band.shape[0]!=k.size: raise ValueError("energies must contain one axis matching k_distance")
    shifted=band-fermi_level; occ=None if occupations is None else np.asarray(occupations,dtype=float)
    if occ is not None:
        if occ.shape!=band.shape and occ.T.shape==band.shape: occ=occ.T
        if occ.shape!=band.shape: raise ValueError("occupations must match energies")
    spin_arr=None if spin is None else np.asarray(spin).reshape(-1)
    if spin_arr is not None and spin_arr.size not in {1,shifted.shape[1]}: raise ValueError("spin must be scalar or one value per band")
    cfg=coerce_config(config,xlabel="Wave vector",ylabel="$E-E_F$ (eV)",ylim=energy_window)
    with theme_context(theme):
        fig,axis,resolved=create_axes(ax=ax,config=cfg,theme=theme); lines=[]
        import matplotlib.colors as mcolors
        cmap_obj=__import__('matplotlib').colormaps['viridis']; occ_mean=None if occ is None else np.nanmean(occ,axis=0); norm=mcolors.Normalize(0,1)
        for index in range(shifted.shape[1]):
            if occ_mean is not None: color=cmap_obj(norm(np.clip(occ_mean[index],0,1))); alpha=0.35+0.65*np.clip(occ_mean[index],0,1)
            else: color=resolved.colors[int(spin_arr[index] if spin_arr is not None and spin_arr.size>1 else index)%2] if spin_arr is not None else resolved.colors[0]; alpha=1.0
            lines.extend(axis.plot(k,shifted[:,index],color=color,linewidth=linewidth,alpha=alpha))
        fermi=axis.axhline(0.0,color="black",linestyle="--",linewidth=0.7); symmetry_lines=[]
        if high_symmetry_positions is not None:
            for position in high_symmetry_positions: symmetry_lines.append(axis.axvline(position,color="0.7",linewidth=0.5))
            if high_symmetry_labels is not None: axis.set_xticks(high_symmetry_positions,high_symmetry_labels)
        colorbar=None
        if occ_mean is not None:
            from matplotlib.cm import ScalarMappable
            colorbar=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap_obj),ax=axis,label="Mean occupation")
        return finalize(fig,axis,config=cfg,artists={"bands":lines,"fermi":fermi,"symmetry":symmetry_lines,"occupation_colorbar":colorbar},data={"k_distance":k,"energies":shifted,"occupations":occ,"spin":spin_arr},theme=theme)


@register_plot(category="materials", aliases=("dos", "density-of-states"))
def density_of_states_plot(
    energy: Any,
    density: Any,
    *,
    components: Mapping[str, Any] | None = None,
    fermi_level: float = 0.0,
    spin_down: Any | None = None,
    fill: bool = True,
    orientation: str = "vertical",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Total and projected density of states with spin mirroring."""
    e = as_array(energy) - fermi_level
    dos = as_array(density)
    cfg = coerce_config(config, xlabel="DOS" if orientation == "horizontal" else "$E-E_F$ (eV)", ylabel="$E-E_F$ (eV)" if orientation == "horizontal" else "DOS")
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        artists = []
        if orientation == "horizontal":
            line = axis.plot(dos, e, label="Total DOS", color="black")[0]
            if fill:
                axis.fill_betweenx(e, 0, dos, color="0.7", alpha=0.25)
            if spin_down is not None:
                axis.plot(-as_array(spin_down), e, color="black")
            if components:
                for index, (name, values) in enumerate(components.items()):
                    artists.extend(axis.plot(as_array(values), e, label=name, color=resolved.colors[index % len(resolved.colors)]))
            axis.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
        else:
            line = axis.plot(e, dos, label="Total DOS", color="black")[0]
            if fill:
                axis.fill_between(e, 0, dos, color="0.7", alpha=0.25)
            if spin_down is not None:
                axis.plot(e, -as_array(spin_down), color="black")
            if components:
                for index, (name, values) in enumerate(components.items()):
                    artists.extend(axis.plot(e, as_array(values), label=name, color=resolved.colors[index % len(resolved.colors)]))
            axis.axvline(0.0, color="black", linestyle="--", linewidth=0.7)
        artists.append(line)
        return finalize(fig, axis, config=cfg, artists={"curves": artists}, data={"energy": e, "density": dos, "components": components})


@register_plot(category="materials", aliases=("band-dos",))
def band_dos_plot(
    k_distance: Any,
    band_energies: Any,
    dos_energy: Any,
    dos: Any,
    *,
    fermi_level: float = 0.0,
    high_symmetry_positions: Sequence[float] | None = None,
    high_symmetry_labels: Sequence[str] | None = None,
    energy_window: tuple[float, float] | None = (-5, 5),
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Journal-style combined band-structure and DOS panel."""
    cfg = coerce_config(config, figsize=(7.0, 4.0), ylim=energy_window)
    with theme_context(theme):
        fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=1, ncols=2, sharey=True, gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})
        band_result = band_structure_plot(k_distance, band_energies, fermi_level=fermi_level, high_symmetry_positions=high_symmetry_positions, high_symmetry_labels=high_symmetry_labels, energy_window=energy_window, ax=axes[0], config=PlotConfig(legend=False, ylabel="$E-E_F$ (eV)"), theme=theme)
        dos_result = density_of_states_plot(dos_energy, dos, fermi_level=fermi_level, orientation="horizontal", ax=axes[1], config=PlotConfig(legend=False, xlabel="DOS", ylim=energy_window), theme=theme)
        axes[1].set_ylabel("")
        return finalize(fig, axes, config=cfg, artists={"band": band_result.artists, "dos": dos_result.artists}, data={"band": band_result.data, "dos": dos_result.data})


@register_plot(category="materials", aliases=("phonons",))
def phonon_dispersion_plot(
    q_distance: Any,
    frequency: Any,
    *,
    high_symmetry_positions: Sequence[float] | None = None,
    high_symmetry_labels: Sequence[str] | None = None,
    imaginary_style: str = "highlight",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Phonon dispersion with imaginary-mode highlighting."""
    q = as_array(q_distance)
    freq = np.asarray(frequency, dtype=float)
    if freq.shape[0] != q.size and freq.shape[1] == q.size:
        freq = freq.T
    cfg = coerce_config(config, xlabel="Wave vector", ylabel="Frequency")
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        lines = []
        for branch in freq.T:
            line = axis.plot(q, branch, color=resolved.colors[0], linewidth=0.8)[0]
            lines.append(line)
            if imaginary_style == "highlight":
                axis.fill_between(q, branch, 0, where=branch < 0, color=resolved.colors[1], alpha=0.25)
        zero = axis.axhline(0.0, color="black", linewidth=0.7)
        symmetry = []
        if high_symmetry_positions is not None:
            for position in high_symmetry_positions:
                symmetry.append(axis.axvline(position, color="0.75", linewidth=0.5))
            if high_symmetry_labels is not None:
                axis.set_xticks(high_symmetry_positions, high_symmetry_labels)
        return finalize(fig, axis, config=cfg, artists={"branches": lines, "zero": zero, "symmetry": symmetry}, data={"q_distance": q, "frequency": freq}, metrics={"minimum_frequency": float(np.nanmin(freq))})


@register_plot(category="materials", aliases=("eos", "equation-of-state"))
def equation_of_state_plot(
    volume: Any,
    energy: Any,
    *,
    fit: bool = True,
    pressure: Any | None = None,
    model: str = "birch-murnaghan",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Energy–volume equation of state with optional pressure overlay."""
    v,e,mask=finite_xy(volume,energy); order=np.argsort(v); v,e=v[order],e[order]
    coefficients=np.polyfit(v,e,min(4,len(v)-1)) if fit and len(v)>2 else None; vv=np.linspace(v.min(),v.max(),300); fitted=np.polyval(coefficients,vv) if coefficients is not None else None
    if coefficients is not None:
        derivative=np.polyder(coefficients); roots=np.roots(derivative); real=roots[np.isreal(roots)].real; candidates=real[(real>=v.min())&(real<=v.max())]; equilibrium_volume=float(candidates[np.argmin(np.polyval(coefficients,candidates))]) if candidates.size else float(v[np.argmin(e)]); equilibrium_energy=float(np.polyval(coefficients,equilibrium_volume)); bulk_modulus=equilibrium_volume*float(np.polyval(np.polyder(coefficients,2),equilibrium_volume)); fitted_pressure=-np.polyval(derivative,vv)
    else:
        equilibrium_volume=float(v[np.argmin(e)]); equilibrium_energy=float(np.min(e)); bulk_modulus=float("nan"); fitted_pressure=None
    pressure_arr=None if pressure is None else np.asarray(pressure,dtype=float).reshape(-1)[mask][order]
    cfg=coerce_config(config,xlabel="Volume",ylabel="Energy")
    with theme_context(theme):
        fig,axis,resolved=create_axes(ax=ax,config=cfg,theme=theme); points=axis.scatter(v,e,label="Calculated"); line=axis.plot(vv,fitted,label=f"{model} fit")[0] if fitted is not None else None; eq=axis.scatter([equilibrium_volume],[equilibrium_energy],marker="*",s=90,color="red",label="$V_0$")
        pressure_axis=None; pressure_points=None; pressure_line=None
        if pressure_arr is not None or fitted_pressure is not None:
            pressure_axis=axis.twinx(); pressure_axis.set_ylabel("Pressure")
            if pressure_arr is not None: pressure_points=pressure_axis.scatter(v,pressure_arr,marker="s",s=16,color=resolved.colors[1],label="Pressure")
            if fitted_pressure is not None: pressure_line=pressure_axis.plot(vv,fitted_pressure,color=resolved.colors[1],linestyle="--",alpha=0.7,label="Fit pressure")[0]
        axis.text(0.04,0.96,f"$V_0$={equilibrium_volume:.4g}\n$B_0$={bulk_modulus:.4g}",transform=axis.transAxes,va="top")
        axes=np.asarray([axis,pressure_axis],dtype=object) if pressure_axis is not None else axis
        return finalize(fig,axes,config=cfg,artists={"points":points,"fit":line,"equilibrium":eq,"pressure_points":pressure_points,"pressure_fit":pressure_line},data={"volume":v,"energy":e,"pressure":pressure_arr,"fit_volume":vv,"fit_energy":fitted,"fit_pressure":fitted_pressure},metrics={"equilibrium_volume":equilibrium_volume,"equilibrium_energy":equilibrium_energy,"bulk_modulus":bulk_modulus},theme=theme)


@register_plot(category="materials", aliases=("stress-strain",))
def stress_strain_plot(
    strain: Any,
    stress: Any,
    *,
    groups: Any | None = None,
    elastic_fit_range: tuple[float, float] | None = None,
    show_toughness: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Stress–strain curves with group-wise modulus, strength and toughness."""
    strain_arr,stress_arr,mask=finite_xy(strain,stress); group_arr=None if groups is None else np.asarray(groups).reshape(-1)[mask]
    cfg=coerce_config(config,xlabel="Strain",ylabel="Stress"); metrics={}; lines=[]; peaks=[]; fills=[]; fits=[]
    with theme_context(theme):
        fig,axis,resolved=create_axes(ax=ax,config=cfg,theme=theme); unique=[None] if group_arr is None else list(dict.fromkeys(group_arr.tolist()))
        for i,group in enumerate(unique):
            selected=np.ones(strain_arr.size,dtype=bool) if group is None else group_arr==group; order=np.argsort(strain_arr[selected]); xs=strain_arr[selected][order]; ys=stress_arr[selected][order]; label="Stress–strain" if group is None else str(group); color=resolved.colors[i%len(resolved.colors)]
            line=axis.plot(xs,ys,label=label,color=color)[0]; lines.append(line); peak_index=int(np.argmax(ys)); peaks.append(axis.scatter([xs[peak_index]],[ys[peak_index]],marker="o",color=color,s=24,label=f"{label} ultimate"))
            toughness=float(np.trapezoid(ys,xs)); modulus=float("nan")
            if show_toughness: fills.append(axis.fill_between(xs,0,ys,color=color,alpha=0.10,label=f"{label} toughness={toughness:.3g}"))
            if elastic_fit_range is not None:
                fit_sel=(xs>=elastic_fit_range[0])&(xs<=elastic_fit_range[1])
                if fit_sel.sum()>=2:
                    coeff=np.polyfit(xs[fit_sel],ys[fit_sel],1); modulus=float(coeff[0]); xfit=np.asarray(elastic_fit_range); fits.append(axis.plot(xfit,np.polyval(coeff,xfit),linestyle="--",color=color,label=f"{label} E={modulus:.3g}")[0])
            prefix="overall" if group is None else str(group); metrics.update({f"{prefix}_modulus":modulus,f"{prefix}_toughness":toughness,f"{prefix}_ultimate_strength":float(ys[peak_index]),f"{prefix}_ultimate_strain":float(xs[peak_index])})
        if group_arr is None: metrics.update({"modulus":metrics["overall_modulus"],"toughness":metrics["overall_toughness"],"ultimate_strength":metrics["overall_ultimate_strength"],"ultimate_strain":metrics["overall_ultimate_strain"]})
        return finalize(fig,axis,config=cfg,artists={"curves":lines,"peaks":peaks,"toughness":fills,"elastic_fits":fits},data={"strain":strain_arr,"stress":stress_arr,"groups":group_arr},metrics=metrics,theme=theme)


@register_plot(category="materials", aliases=("xrd", "diffraction-pattern"))
def diffraction_pattern_plot(
    two_theta: Any,
    intensity: Any,
    *,
    references: Mapping[str, tuple[Any, Any]] | None = None,
    normalize: bool = True,
    log_intensity: bool = False,
    annotate_peaks: int = 0,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """XRD/neutron diffraction pattern with reference sticks and peak labels."""
    angle, intensity_arr, _ = finite_xy(two_theta, intensity)
    if normalize:
        intensity_arr = intensity_arr / max(np.nanmax(intensity_arr), 1e-15) * 100.0
    cfg = coerce_config(config, xlabel=r"$2\theta$ (deg)", ylabel="Intensity" + (" (%)" if normalize else ""), yscale="log" if log_intensity else None)
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        line = axis.plot(angle, intensity_arr, label="Pattern")[0]
        reference_artists = []
        if references:
            baseline = np.nanmin(intensity_arr)
            for index, (name, (positions, heights)) in enumerate(references.items()):
                pos = as_array(positions)
                h = as_array(heights)
                if normalize:
                    h = h / max(np.max(h), 1e-15) * np.max(intensity_arr) * 0.25
                artist = axis.vlines(pos, baseline, baseline + h, color=resolved.colors[(index + 1) % len(resolved.colors)], label=name, linewidth=0.8)
                reference_artists.append(artist)
        if annotate_peaks > 0:
            try:
                from scipy.signal import find_peaks

                peaks, properties = find_peaks(intensity_arr, prominence=np.ptp(intensity_arr) * 0.02)
                selected = peaks[np.argsort(intensity_arr[peaks])[-annotate_peaks:]]
                for peak in selected:
                    axis.annotate(f"{angle[peak]:.2f}°", (angle[peak], intensity_arr[peak]), xytext=(0, 5), textcoords="offset points", ha="center", rotation=90, fontsize="x-small")
            except Exception:
                pass
        return finalize(fig, axis, config=cfg, artists={"pattern": line, "references": reference_artists}, data={"two_theta": angle, "intensity": intensity_arr})


@register_plot(category="materials", aliases=("spectra-stack", "waterfall-spectra"))
def stacked_spectra(
    x: Any,
    spectra: Any,
    *,
    labels: Sequence[str] | None = None,
    offsets: float | Sequence[float] = 1.0,
    normalize: bool = False,
    fill: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Stack Raman/IR/XPS/NMR/XAS spectra with controlled vertical offsets."""
    x_arr = as_array(x)
    matrix = np.asarray(spectra, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[1] != x_arr.size and matrix.shape[0] == x_arr.size:
        matrix = matrix.T
    if normalize:
        matrix = matrix / np.maximum(np.nanmax(np.abs(matrix), axis=1, keepdims=True), 1e-15)
    offset_arr = np.arange(matrix.shape[0]) * float(offsets) if np.isscalar(offsets) else as_array(offsets)
    names = make_labels(labels, matrix.shape[0], "Spectrum")
    cfg = coerce_config(config)
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        lines = []
        for index, (row, offset, name) in enumerate(zip(matrix, offset_arr, names)):
            shifted = row + offset
            line = axis.plot(x_arr, shifted, label=name, color=resolved.colors[index % len(resolved.colors)])[0]
            lines.append(line)
            if fill:
                axis.fill_between(x_arr, offset, shifted, color=line.get_color(), alpha=0.15)
            axis.text(x_arr[-1], shifted[-1], name, va="center", ha="left", fontsize="small")
        return finalize(fig, axis, config=cfg, artists={"spectra": lines}, data={"x": x_arr, "spectra": matrix, "offsets": offset_arr, "labels": names})


@register_plot(category="materials", aliases=("elastic-polar", "directional-property"))
def elastic_polar_plot(
    angle: Any,
    values: Any,
    *,
    labels: Sequence[str] | None = None,
    fill_alpha: float = 0.1,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Polar anisotropy plot for modulus, conductivity or surface energy."""
    theta = as_array(angle)
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    names = make_labels(labels, matrix.shape[0], "Property")
    cfg = coerce_config(config, figsize=(5.0, 5.0))
    with theme_context(theme):
        fig, axis, _ = create_axes(config=cfg, theme=theme, projection="polar")
        lines = []
        for row, name in zip(matrix, names):
            line = axis.plot(theta, row, label=name)[0]
            axis.fill(theta, row, color=line.get_color(), alpha=fill_alpha)
            lines.append(line)
        return finalize(fig, axis, config=cfg, artists={"curves": lines}, data={"angle": theta, "values": matrix, "labels": names})
