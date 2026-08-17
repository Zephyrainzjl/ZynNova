from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ._core import PlotResult, get_theme, theme_context
from ._registry import register_plot


@dataclass(slots=True)
class AnimationResult:
    animation: Any
    figure: Any
    axes: Any
    metadata: dict[str, Any]

    def display(self) -> Any:
        try:
            from IPython.display import HTML

            return HTML(self.animation.to_jshtml())
        except Exception:
            return self.animation

    def save(self, path: str | Path, *, fps: int = 15, dpi: int = 120, writer: str | None = None) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix.lower()
        if writer is None:
            writer = "pillow" if suffix == ".gif" else "ffmpeg"
        self.animation.save(target, writer=writer, fps=fps, dpi=dpi)
        return target


@register_plot(category="animation", aliases=("animate-line", "trajectory-animation"))
def animate_series(
    x: Any,
    frames: Any,
    *,
    labels: Sequence[str] | None = None,
    interval_ms: int = 60,
    repeat: bool = True,
    trail: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | Callable[[int], str] | None = None,
    theme: Any = None,
) -> AnimationResult:
    """Animate one or multiple evolving 1-D result series."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    x_arr=np.asarray(x,dtype=float); data=np.asarray(frames,dtype=float)
    if data.ndim==2: data=data[:,None,:]
    if data.ndim!=3 or data.shape[2]!=x_arr.size: raise ValueError("frames must have shape (time, series, x)")
    names=[f"Series {i+1}" for i in range(data.shape[1])] if labels is None else list(labels)
    if len(names)!=data.shape[1]: raise ValueError("labels must match series")
    with theme_context(theme):
        fig,axis=plt.subplots(); lines=[axis.plot(x_arr,data[0,i],label=names[i])[0] for i in range(data.shape[1])]
        trail_lines=[]
        if trail:
            for line in lines:
                trail_lines.append([axis.plot([],[],color=line.get_color(),alpha=0.08+0.06*j,linewidth=0.7)[0] for j in range(6)])
        axis.set_xlim(*(xlim or (x_arr.min(),x_arr.max()))); axis.set_ylim(*(ylim or (np.nanmin(data),np.nanmax(data)))); axis.legend()
        def update(frame:int):
            artists=list(lines)
            for i,line in enumerate(lines):
                line.set_ydata(data[frame,i])
                if trail:
                    for j,tline in enumerate(trail_lines[i],start=1):
                        previous=frame-j
                        if previous>=0: tline.set_data(x_arr,data[previous,i])
                        else: tline.set_data([],[])
                        artists.append(tline)
            if title is not None: axis.set_title(title(frame) if callable(title) else f"{title} — frame {frame}")
            return artists
        animation=FuncAnimation(fig,update,frames=data.shape[0],interval=interval_ms,repeat=repeat,blit=False)
        return AnimationResult(animation,fig,axis,{"frame_count":data.shape[0],"kind":"series","trail":bool(trail)})


@register_plot(category="animation", aliases=("animate-field", "field-animation"))
def animate_field(
    frames: Any,
    *,
    times: Any | None = None,
    interval_ms: int = 60,
    repeat: bool = True,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str = "Value",
    title: str | None = None,
    theme: Any = None,
) -> AnimationResult:
    """Animate 2-D scalar fields, phase fields, microscopy or spatial maps."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    data = np.asarray(frames, dtype=float)
    if data.ndim != 3:
        raise ValueError("frames must have shape (time, y, x)")
    time_arr = np.arange(data.shape[0]) if times is None else np.asarray(times)
    with theme_context(theme):
        fig, axis = plt.subplots()
        image = axis.imshow(data[0], origin="lower", cmap=cmap, vmin=np.nanmin(data) if vmin is None else vmin, vmax=np.nanmax(data) if vmax is None else vmax, animated=True)
        colorbar = fig.colorbar(image, ax=axis, label=colorbar_label)
        axis.set_axis_off()
        def update(frame: int):
            image.set_data(data[frame])
            axis.set_title(f"{title + ' — ' if title else ''}t={time_arr[frame]:.4g}")
            return [image]
        animation = FuncAnimation(fig, update, frames=data.shape[0], interval=interval_ms, repeat=repeat, blit=False)
        return AnimationResult(animation, fig, axis, {"frame_count": data.shape[0], "kind": "field", "colorbar": colorbar})


@register_plot(category="animation", aliases=("animate-embedding",))
def animate_embedding(
    embeddings: Any,
    *,
    color: Any | None = None,
    labels: Any | None = None,
    interval_ms: int = 80,
    trail: int = 0,
    cmap: str = "viridis",
    theme: Any = None,
) -> AnimationResult:
    """Animate evolving embeddings, latent trajectories or particle projections."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    data=np.asarray(embeddings,dtype=float)
    if data.ndim!=3 or data.shape[2]<2: raise ValueError("embeddings must have shape (time, samples, >=2)")
    scalar=np.arange(data.shape[1]) if color is None else np.asarray(color).reshape(-1)
    if scalar.size!=data.shape[1]: raise ValueError("color must match samples")
    label_arr=None if labels is None else np.asarray(labels).reshape(-1)
    if label_arr is not None and label_arr.size!=data.shape[1]: raise ValueError("labels must match samples")
    with theme_context(theme):
        fig,axis=plt.subplots(); points=axis.scatter(data[0,:,0],data[0,:,1],c=scalar,cmap=cmap,s=10,alpha=0.75)
        trail_artist=axis.scatter([],[],c=[],cmap=cmap,s=5,alpha=0.15) if trail>0 else None
        texts=[]
        if label_arr is not None:
            texts=[axis.text(data[0,i,0],data[0,i,1],str(label_arr[i]),fontsize=6) for i in range(data.shape[1])]
        axis.set_xlim(np.nanmin(data[...,0]),np.nanmax(data[...,0])); axis.set_ylim(np.nanmin(data[...,1]),np.nanmax(data[...,1]))
        def update(frame:int):
            points.set_offsets(data[frame,:,:2]); artists=[points]
            if trail_artist is not None:
                start=max(0,frame-int(trail)); history=data[start:frame].reshape(-1,data.shape[2]) if frame>start else np.empty((0,data.shape[2]))
                trail_artist.set_offsets(history[:,:2] if history.size else np.empty((0,2)))
                repeated=np.tile(scalar,max(frame-start,0)); trail_artist.set_array(repeated.astype(float) if repeated.size else np.asarray([],dtype=float)); artists.append(trail_artist)
            for i,text in enumerate(texts): text.set_position(data[frame,i,:2]); artists.append(text)
            axis.set_title(f"Frame {frame}"); return artists
        animation=FuncAnimation(fig,update,frames=data.shape[0],interval=interval_ms,repeat=True,blit=False)
        return AnimationResult(animation,fig,axis,{"frame_count":data.shape[0],"kind":"embedding","trail":int(trail),"labels":label_arr})
