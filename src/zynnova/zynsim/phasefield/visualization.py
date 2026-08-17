"""Live and recorded dynamic visualization for 1D, 2D, and 3D phase fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fields import PhaseFieldResult, PhaseFieldState, PhaseFieldTrajectory


def _trajectory(source: PhaseFieldResult | PhaseFieldTrajectory) -> PhaseFieldTrajectory:
    return source.trajectory if isinstance(source, PhaseFieldResult) else source


@dataclass(slots=True)
class PhaseFieldAnimator:
    """Matplotlib animation that automatically adapts to 1D, 2D, or 3D data."""

    source: PhaseFieldResult | PhaseFieldTrajectory
    field_name: str
    interval_ms: int = 80
    cmap: str = "viridis"
    value_range: tuple[float, float] | None = None
    slice_indices: tuple[int, int, int] | None = None
    title: str | None = None

    def _range(self, stack: np.ndarray) -> tuple[float, float]:
        if self.value_range is not None:
            return self.value_range
        return float(np.nanmin(stack)), float(np.nanmax(stack))

    def animation(self, *, repeat: bool = True, blit: bool = False):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        trajectory = _trajectory(self.source)
        if not trajectory.frames:
            raise ValueError("cannot animate an empty phase-field trajectory")
        stack = trajectory.stack(self.field_name)
        grid = trajectory.frames[0].grid
        vmin, vmax = self._range(stack)
        heading = self.title or self.field_name

        if grid.dimensions == 1:
            x = grid.coordinates()
            figure, axis = plt.subplots(figsize=(8, 4))
            (line,) = axis.plot(x, stack[0])
            axis.set_xlim(float(x.min()), float(x.max()))
            margin = 0.05 * max(vmax - vmin, 1.0e-12)
            axis.set_ylim(vmin - margin, vmax + margin)
            axis.set_xlabel("Position")
            axis.set_ylabel(self.field_name)
            time_text = axis.set_title(f"{heading} — t={trajectory.frames[0].time:.4g}")

            def update(index):
                line.set_ydata(stack[index])
                time_text.set_text(f"{heading} — t={trajectory.frames[index].time:.4g}")
                return line, time_text

            artists = (line, time_text)

        elif grid.dimensions == 2:
            figure, axis = plt.subplots(figsize=(6, 5))
            image = axis.imshow(
                stack[0],
                origin="lower",
                interpolation="nearest",
                cmap=self.cmap,
                vmin=vmin,
                vmax=vmax,
                extent=(0.0, grid.lengths[1], 0.0, grid.lengths[0]),
                aspect="auto",
            )
            figure.colorbar(image, ax=axis, label=self.field_name)
            time_text = axis.set_title(f"{heading} — t={trajectory.frames[0].time:.4g}")
            axis.set_xlabel("x")
            axis.set_ylabel("y")

            def update(index):
                image.set_data(stack[index])
                time_text.set_text(f"{heading} — t={trajectory.frames[index].time:.4g}")
                return image, time_text

            artists = (image, time_text)

        else:
            shape = grid.shape
            indices = self.slice_indices or (shape[0] // 2, shape[1] // 2, shape[2] // 2)
            z_index, y_index, x_index = indices
            figure, axes = plt.subplots(1, 3, figsize=(13, 4))
            images = [
                axes[0].imshow(stack[0, z_index, :, :], origin="lower", cmap=self.cmap, vmin=vmin, vmax=vmax),
                axes[1].imshow(stack[0, :, y_index, :], origin="lower", cmap=self.cmap, vmin=vmin, vmax=vmax),
                axes[2].imshow(stack[0, :, :, x_index], origin="lower", cmap=self.cmap, vmin=vmin, vmax=vmax),
            ]
            axes[0].set_title(f"z={z_index}")
            axes[1].set_title(f"y={y_index}")
            axes[2].set_title(f"x={x_index}")
            for axis in axes:
                axis.set_axis_off()
            figure.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.78, label=self.field_name)
            time_text = figure.suptitle(f"{heading} — t={trajectory.frames[0].time:.4g}")

            def update(index):
                images[0].set_data(stack[index, z_index, :, :])
                images[1].set_data(stack[index, :, y_index, :])
                images[2].set_data(stack[index, :, :, x_index])
                time_text.set_text(f"{heading} — t={trajectory.frames[index].time:.4g}")
                return (*images, time_text)

            artists = (*images, time_text)

        animation = FuncAnimation(
            figure,
            update,
            frames=len(trajectory.frames),
            interval=self.interval_ms,
            repeat=repeat,
            blit=blit,
        )
        animation._draw_was_started = True  # avoid false warning in notebook workflows
        return animation

    def display(self):
        """Display an interactive JavaScript animation inside Jupyter."""

        from IPython.display import HTML, display

        animation = self.animation()
        html = HTML(animation.to_jshtml())
        display(html)
        return html

    def save(self, path: str | Path, *, fps: int = 15, dpi: int = 120) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        animation = self.animation()
        suffix = target.suffix.lower()
        if suffix == ".gif":
            animation.save(target, writer="pillow", fps=fps, dpi=dpi)
        elif suffix in {".mp4", ".m4v"}:
            animation.save(target, writer="ffmpeg", fps=fps, dpi=dpi)
        elif suffix == ".html":
            target.write_text(animation.to_jshtml(), encoding="utf-8")
        else:
            raise ValueError("animation path must end in .gif, .mp4, .m4v, or .html")
        return target


class LivePhaseFieldViewer:
    """Solver callback that updates a Matplotlib figure while the phase field moves."""

    def __init__(
        self,
        field_name: str,
        *,
        update_every: int = 1,
        cmap: str = "viridis",
        value_range: tuple[float, float] | None = None,
        slice_indices: tuple[int, int, int] | None = None,
        clear_output: bool = False,
    ) -> None:
        if update_every <= 0:
            raise ValueError("update_every must be positive")
        self.field_name = field_name
        self.update_every = update_every
        self.cmap = cmap
        self.value_range = value_range
        self.slice_indices = slice_indices
        self.clear_output = clear_output
        self._figure = None
        self._artists: list[Any] = []
        self._display_handle = None

    def _initialize(self, state: PhaseFieldState) -> None:
        import matplotlib.pyplot as plt

        values = np.asarray(state.fields[self.field_name])
        vmin, vmax = self.value_range or (float(values.min()), float(values.max()))
        if state.grid.dimensions == 1:
            self._figure, axis = plt.subplots(figsize=(8, 4))
            x = state.grid.coordinates()
            (line,) = axis.plot(x, values)
            axis.set_ylim(vmin, vmax if vmax > vmin else vmin + 1.0)
            self._artists = [line, axis]
        elif state.grid.dimensions == 2:
            self._figure, axis = plt.subplots(figsize=(6, 5))
            image = axis.imshow(values, origin="lower", cmap=self.cmap, vmin=vmin, vmax=vmax)
            self._figure.colorbar(image, ax=axis, label=self.field_name)
            self._artists = [image, axis]
        else:
            shape = state.grid.shape
            z, y, x = self.slice_indices or (shape[0] // 2, shape[1] // 2, shape[2] // 2)
            self._figure, axes = plt.subplots(1, 3, figsize=(13, 4))
            images = [
                axes[0].imshow(values[z], origin="lower", cmap=self.cmap, vmin=vmin, vmax=vmax),
                axes[1].imshow(values[:, y, :], origin="lower", cmap=self.cmap, vmin=vmin, vmax=vmax),
                axes[2].imshow(values[:, :, x], origin="lower", cmap=self.cmap, vmin=vmin, vmax=vmax),
            ]
            self._artists = [*images, axes, (z, y, x)]
        self._figure.suptitle(f"{self.field_name} — t={state.time:.4g}")
        self._figure.tight_layout()

    def __call__(self, state: PhaseFieldState, diagnostics=None) -> None:
        del diagnostics
        if state.step % self.update_every != 0:
            return
        import matplotlib.pyplot as plt

        if self._figure is None:
            self._initialize(state)
        values = np.asarray(state.fields[self.field_name])
        if state.grid.dimensions == 1:
            self._artists[0].set_ydata(values)
        elif state.grid.dimensions == 2:
            self._artists[0].set_data(values)
        else:
            z, y, x = self._artists[-1]
            self._artists[0].set_data(values[z])
            self._artists[1].set_data(values[:, y, :])
            self._artists[2].set_data(values[:, :, x])
        self._figure.suptitle(f"{self.field_name} — t={state.time:.4g}")
        self._figure.canvas.draw_idle()
        try:
            from IPython.display import clear_output, display

            if self.clear_output:
                clear_output(wait=True)
            if self._display_handle is None:
                self._display_handle = display(self._figure, display_id=True)
            else:
                self._display_handle.update(self._figure)
        except (ImportError, AttributeError):
            plt.pause(0.001)

    def close(self) -> None:
        if self._figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self._figure)


@dataclass(slots=True)
class PyVistaPhaseFieldViewer:
    """Interactive 3D volume/isosurface playback and GIF export using PyVista."""

    source: PhaseFieldResult | PhaseFieldTrajectory
    field_name: str
    iso_value: float = 0.0
    volume: bool = False

    def _pyvista(self):
        try:
            import pyvista as pv
        except ImportError as exc:
            raise RuntimeError("PyVista is required for interactive 3D phase-field rendering") from exc
        return pv

    def show(self, *, notebook: bool | None = None):
        pv = self._pyvista()
        trajectory = _trajectory(self.source)
        frame = trajectory.frames[0]
        if frame.grid.dimensions != 3:
            raise ValueError("PyVistaPhaseFieldViewer requires a three-dimensional trajectory")
        values = np.asarray(frame.fields[self.field_name])
        grid = pv.ImageData(
            dimensions=np.asarray(frame.grid.shape) + 1,
            spacing=frame.grid.spacing,
            origin=frame.grid.origin,
        )
        cell_values = np.asarray(values).ravel(order="F")
        grid.cell_data[self.field_name] = cell_values
        plotter = pv.Plotter(notebook=notebook)
        if self.volume:
            plotter.add_volume(grid, scalars=self.field_name)
        else:
            plotter.add_mesh(grid.contour([self.iso_value], scalars=self.field_name), scalars=self.field_name)
        plotter.add_text(f"{self.field_name}, t={frame.time:.4g}")
        return plotter.show()

    def save_gif(self, path: str | Path, *, fps: int = 12) -> Path:
        pv = self._pyvista()
        trajectory = _trajectory(self.source)
        first = trajectory.frames[0]
        if first.grid.dimensions != 3:
            raise ValueError("PyVista GIF export requires a 3D trajectory")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        image = pv.ImageData(
            dimensions=np.asarray(first.grid.shape) + 1,
            spacing=first.grid.spacing,
            origin=first.grid.origin,
        )
        image.cell_data[self.field_name] = np.asarray(first.fields[self.field_name]).ravel(order="F")
        plotter = pv.Plotter(off_screen=True)
        actor = plotter.add_mesh(image.contour([self.iso_value], scalars=self.field_name))
        plotter.open_gif(str(target), fps=fps)
        for frame in trajectory.frames:
            image.cell_data[self.field_name] = np.asarray(frame.fields[self.field_name]).ravel(order="F")
            contour = image.contour([self.iso_value], scalars=self.field_name)
            actor.mapper.dataset.copy_from(contour)
            plotter.write_frame()
        plotter.close()
        return target


def animate_phase_field(
    source: PhaseFieldResult | PhaseFieldTrajectory,
    field_name: str,
    **kwargs: Any,
):
    """Create a dimension-aware Matplotlib animation."""

    return PhaseFieldAnimator(source, field_name, **kwargs).animation()


__all__ = [
    "LivePhaseFieldViewer",
    "PhaseFieldAnimator",
    "PyVistaPhaseFieldViewer",
    "animate_phase_field",
]
