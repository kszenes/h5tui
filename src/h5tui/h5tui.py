from textual import on, _widget_navigation
from textual.app import App, ComposeResult
from textual.events import Resize
from textual.reactive import Reactive
from textual.widgets import (
    Footer,
    Header,
    OptionList,
    Static,
    DataTable,
    Select,
    Checkbox,
    Input,
    Button,
)
from textual.containers import (
    VerticalScroll,
    Horizontal,
    Vertical,
    VerticalGroup,
    HorizontalGroup,
)
from textual.binding import Binding
from textual.geometry import clamp
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.content import Content
from textual_plotext import PlotextPlot
from rich.markup import escape
from rich.text import Text

import h5py
import numpy as np

import sys
import os
import re
import argparse
from collections import defaultdict

UNICODE_SUPPORT = sys.stdout.encoding.lower().startswith("utf")

# Compound 1D plots: X axis option meaning row index (0..n-1), not a dtype field.
COMPOUND_PLOT_INDEX = "__index__"

# NumPy ndarray text formatting (explicit kwargs; do not rely on np.set_printoptions).
_ND_THRESHOLD = 1000
_ND_EDGEITEMS = 3
_ND_LINEWIDTH_DEFAULT = 75


def h5_child_path(parent: str, name: str) -> str:
    """Join HDF5 posix path `parent` with child `name`."""
    if parent == "/":
        return f"/{name}"
    return f"{parent}/{name}"


def h5_parent_path(path: str) -> str | None:
    """Parent HDF5 group path, or None if `path` is the file root."""
    if not path or path == "/":
        return None
    parent = os.path.dirname(path)
    if parent == path:
        return None
    return parent or "/"


def _build_link_canonical_name_map(group: h5py.Group, dir_path: str) -> dict[str, str]:
    """For names in ``group`` that share an HDF5 object id, map alias name -> canonical path.

    Soft and external links are skipped: they are identified by ``_link_arrow_suffix`` instead.
    """
    id_to_names: defaultdict[object, list[str]] = defaultdict(list)
    for name in group.keys():
        try:
            lnk = group.get(name, getlink=True)
        except Exception:
            continue
        if isinstance(lnk, (h5py.SoftLink, h5py.ExternalLink)):
            continue
        try:
            obj = group[name]
        except Exception:
            continue
        id_to_names[obj.id].append(name)
    out: dict[str, str] = {}
    for names in id_to_names.values():
        if len(names) < 2:
            continue
        full_paths = {n: h5_child_path(dir_path, n) for n in names}
        canonical = min(full_paths.values(), key=lambda p: (len(p), p))
        for n, fp in full_paths.items():
            if fp != canonical:
                out[n] = canonical
    return out


def _format_external_link_target(lnk: h5py.ExternalLink) -> str:
    fn = os.path.basename(lnk.filename) or lnk.filename
    return f"{fn}:{lnk.path}"


def _link_arrow_suffix(
    group: h5py.Group,
    dir_path: str,
    name: str,
    canonical_map: dict[str, str],
) -> str:
    """Yazi-style `` -> target`` for soft, external, and hard-link aliases."""
    try:
        lnk = group.get(name, getlink=True)
    except Exception:
        return ""
    if isinstance(lnk, h5py.SoftLink):
        return f" -> {lnk.path}"
    if isinstance(lnk, h5py.ExternalLink):
        return f" -> {_format_external_link_target(lnk)}"
    if name in canonical_map:
        return f" -> {canonical_map[name]}"
    return ""


def _h5_child_link_blocked_reason(
    h5file: h5py.File, dir_path: str, name: str
) -> str | None:
    """If the child cannot be opened (broken soft/external link), return a message; else ``None``.

    Uses membership checks — no reliance on opening a broken soft link.
    """
    parent = h5file[dir_path]
    if not isinstance(parent, h5py.Group):
        return None
    try:
        lnk = parent.get(name, getlink=True)
    except Exception:
        return None
    if isinstance(lnk, h5py.SoftLink):
        if lnk.path not in h5file:
            return "Broken soft link (target is missing)"
        return None
    if isinstance(lnk, h5py.ExternalLink):
        if not os.path.isfile(lnk.filename):
            return "External link: file not found"
        if not h5py.is_hdf5(lnk.filename):
            return "External link: not a valid HDF5 file"
        try:
            with h5py.File(lnk.filename, "r") as ext:
                if lnk.path not in ext:
                    return "Broken external link (target is missing)"
        except OSError as e:
            return f"External link: cannot open file ({e})"
    return None


def _format_nbytes_human(n: int) -> str:
    """Format a byte count using B, KB, MB, GB, TB (1024-based)."""
    if n < 0:
        return str(n)
    x = float(n)
    units = ("B", "KB", "MB", "GB", "TB")
    for i, u in enumerate(units):
        if x < 1024.0 or u == "TB":
            if u == "B":
                return f"{int(n)} B"
            if abs(x - round(x)) < 1e-9:
                return f"{int(round(x))} {u}"
            return f"{x:.1f} {u}"
        x /= 1024.0
    return f"{int(n)} B"


def _dataset_size_display(dset: h5py.Dataset) -> str:
    """Human-readable stored size (``nbytes``), or ``unknown`` if not available."""
    try:
        return _format_nbytes_human(int(dset.nbytes))
    except Exception:
        return "unknown"


def _dataset_compression_detail(dset: h5py.Dataset) -> str | None:
    """``None`` if uncompressed; else ``gzip(4)``-style label."""
    comp = dset.compression
    if not comp:
        return None
    opts = dset.compression_opts
    if opts is not None and opts != ():
        return f"{comp}({opts})"
    return str(comp)


def _structured_field_count(dset: h5py.Dataset) -> int:
    n = dset.dtype.names
    return len(n) if n else 0


def _structured_dataset_summary_line(dset: h5py.Dataset) -> str:
    """Single-line summary for structured datasets (same wording as header / browse preview)."""
    nf = _structured_field_count(dset)
    sh = dset.shape
    if dset.ndim == 0:
        return f"structured scalar · {nf} fields"
    if dset.ndim == 1:
        return f"{int(sh[0])} rows × {nf} fields"
    if dset.ndim == 2:
        return f"{int(sh[0])} × {int(sh[1])} · {nf} fields/element"
    return f"structured {sh} · {nf} fields/element"


def _plain_dataset_shape_display(dset: h5py.Dataset) -> str:
    """Shape for plain (non-structured) datasets; 2D uses ``nrows × ncols`` (Unicode ×)."""
    sh = dset.shape
    if dset.ndim == 2:
        return f"{int(sh[0])} × {int(sh[1])}"
    return str(sh)


def format_dataset_header_metadata(dset: h5py.Dataset) -> str:
    """Build the Dataset header line: dtype + shape, or compact structured summary."""
    if dset.dtype.fields is not None:
        parts = [_structured_dataset_summary_line(dset)]
    else:
        dtype_display = str(dset.dtype)
        parts = [f"<{dtype_display}> {_plain_dataset_shape_display(dset)}"]

    if dset.chunks is not None:
        parts.append(f"chunks={dset.chunks}")

    cd = _dataset_compression_detail(dset)
    if cd:
        parts.append(f"compression={cd}")

    filt: list[str] = []
    if dset.shuffle:
        filt.append("shuffle")
    if dset.fletcher32:
        filt.append("fletcher32")
    if filt:
        parts.append("filters=" + "+".join(filt))

    if dset.scaleoffset is not None:
        parts.append(f"scale-offset={dset.scaleoffset!r}")

    ms = getattr(dset, "maxshape", None)
    if ms is not None and ms != dset.shape:
        parts.append(f"maxshape={ms}")

    if getattr(dset, "is_virtual", False):
        parts.append("virtual")
        try:
            parts.append(f"vds_sources={len(dset.virtual_sources())}")
        except Exception:
            pass

    parts.append(f"{_dataset_size_display(dset)}")

    return " | ".join(parts)


class AttributeScreen(ModalScreen):
    BINDINGS = [
        Binding(
            "left,h,escape,q",
            "quit_attrs",
            "Close",
            show=True,
            priority=True,
        ),
        Binding("Q", "quit_application", "Quit app", show=True, priority=True),
        Binding("down,j", "cursor_down", "Down", show=True, priority=True),
        Binding("up,k", "cursor_up", "Up", show=True, priority=True),
        Binding("J", "scroll_content_down", "Scroll Down", priority=True),
        Binding("K", "scroll_content_up", "Scroll Up", priority=True),
        Binding("u", "scroll_content_page_up", "Page Up", priority=True),
        Binding("d", "scroll_content_page_down", "Page Down", priority=True),
    ]

    def __init__(self, h5file, cur_dir, itemname, id=None) -> None:
        super().__init__(id=id)
        self._file = h5file
        self._cur_dir = cur_dir
        self._itemname = itemname
        self._item = self._file[h5_child_path(self._cur_dir, self._itemname)]
        self._attrs = list(self._item.attrs.keys())

        self._cur_attr = self._attrs[0]

        self._selector_widget = MyOptionList(*self._attrs, markup=False)
        self._selector_widget.border_title = f"Attributes for {self._itemname}"

        self._content_widget = Static(id="attr_content", markup=False)
        self._vertical_widget = VerticalScroll(
            self._content_widget, id="attr_content_scroll"
        )

        self.update_content()

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield self._selector_widget
            yield self._vertical_widget
        yield Footer()

    def update_content(self):
        content = str(self._item.attrs[self._cur_attr])
        self._content_widget.update(content)

    def action_quit_attrs(self):
        self.app.pop_screen()

    def action_quit_application(self) -> None:
        self.app.exit()

    def action_cursor_down(self):
        self._selector_widget.action_cursor_down()
        highlighted = self._selector_widget.highlighted
        if highlighted is not None:
            self._cur_attr = self._attrs[highlighted]
            self.update_content()

    def action_cursor_up(self):
        self._selector_widget.action_cursor_up()
        highlighted = self._selector_widget.highlighted
        if highlighted is not None:
            self._cur_attr = self._attrs[highlighted]
            self.update_content()

    def action_scroll_content_down(self):
        self._vertical_widget.scroll_down()

    def action_scroll_content_up(self):
        self._vertical_widget.scroll_up()

    def action_scroll_content_page_down(self):
        self._vertical_widget.scroll_page_down()

    def action_scroll_content_page_up(self):
        self._vertical_widget.scroll_page_up()


def _default_histogram_bins(data: np.ndarray) -> int:
    """Heuristic bin count (matches previous default behavior)."""
    vals = np.asarray(data, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 10
    n = int(vals.size)
    return min(40, max(10, int(np.sqrt(n))))


def _plot_1d_histogram(
    plot_plt,
    data: np.ndarray,
    bins: int | None = None,
    *,
    title: str | None = None,
) -> None:
    """Draw a histogram of ``data`` (1D) using NumPy binning and Plotext ``bar``."""
    vals = np.asarray(data, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        if title:
            plot_plt.title(title)
        plot_plt.xlabel("Value")
        plot_plt.ylabel("Count")
        return
    nb = _default_histogram_bins(data) if bins is None else max(1, int(bins))
    counts, edges = np.histogram(vals, bins=nb)
    x = ((edges[:-1] + edges[1:]) / 2).tolist()
    y = counts.astype(float).tolist()
    plot_plt.bar(x, y, color="cyan")
    if title:
        plot_plt.title(title)
    plot_plt.xlabel("Value")
    plot_plt.ylabel("Count")


_PLOT_2D_HEATMAP_THRESHOLD = 100


def _plotext_reset_linear(plot_plt) -> None:
    plot_plt.clear_figure()
    plot_plt.xscale("linear")
    plot_plt.yscale("linear")


def _plot_preview_compound_1d(
    plot_plt,
    data: np.ndarray,
    *,
    histogram: bool,
    compound_x: str,
    compound_y: str,
    log_x: bool,
    log_y: bool,
    histogram_value_field: str | None = None,
) -> None:
    if histogram:
        field = (
            histogram_value_field
            if histogram_value_field is not None
            else str(compound_y)
        )
        _plotext_reset_linear(plot_plt)
        _plot_1d_histogram(
            plot_plt,
            np.asarray(data[field], dtype=np.float64),
            title=str(field),
        )
        return
    _plotext_compound_line(
        plot_plt,
        data,
        x_key=str(compound_x),
        y_key=str(compound_y),
        log_x=log_x,
        log_y=log_y,
    )


def _plot_preview_plain_1d(
    plot_plt,
    y,
    *,
    histogram: bool,
    x: np.ndarray | None = None,
) -> None:
    y = np.asarray(y)
    if histogram:
        _plot_1d_histogram(plot_plt, y, title="Values")
        return
    xv = np.arange(y.shape[0]) if x is None else np.asarray(x)
    plot_plt.title("Value vs Index")
    plot_plt.xlabel("Index")
    plot_plt.plot(xv, y, color="cyan", marker="braille")


class _PlotextHeatmapDataFrame:
    """Minimal pandas.DataFrame stand-in for plotext ``heatmap`` (``plotext._monitor.draw_heatmap``)."""

    __slots__ = ("columns", "index", "values")

    def __init__(self, data: np.ndarray) -> None:
        data = np.asarray(data)
        nrows, ncols = data.shape
        self.columns = np.arange(ncols)
        self.index = np.arange(nrows)
        self.values = data

    def __repr__(self) -> str:
        return np.array2string(self.values, max_line_width=120)


def _plot_preview_2d(plot_plt, data: np.ndarray) -> None:
    data = np.asarray(data)
    nrows, ncols = data.shape
    plot_plt.plot_size(nrows, ncols)
    if nrows < _PLOT_2D_HEATMAP_THRESHOLD and ncols < _PLOT_2D_HEATMAP_THRESHOLD:
        plot_plt.heatmap(_PlotextHeatmapDataFrame(data))
        plot_plt.title("")
    else:
        plot_plt.matrix_plot(data.tolist())
    plot_plt.xlabel("Column")
    plot_plt.ylabel("Row")


def _cubic_smooth_xy(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return dense (x, y) for a cubic spline through finite samples, or ``None``."""
    from scipy.interpolate import CubicSpline

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return None
    m = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[m], y[m]
    if xv.size < 2:
        return None
    order = np.argsort(xv)
    xv, yv = xv[order], yv[order]
    # collapse duplicate x (rare for index axis)
    if np.any(np.diff(xv) == 0):
        uniq = np.concatenate([[True], np.diff(xv) != 0])
        xv, yv = xv[uniq], yv[uniq]
    if xv.size < 2:
        return None
    x_fine = np.linspace(float(xv[0]), float(xv[-1]), max(64, int(xv.size) * 8))
    if xv.size >= 4:
        y_fine = CubicSpline(xv, yv)(x_fine)
    else:
        y_fine = np.interp(x_fine, xv, yv)
    return x_fine, y_fine


def _open_saved_plot_path(path: str) -> None:
    """Open ``path`` with the OS default application (e.g. PDF viewer)."""
    import os
    import shutil
    import subprocess
    import sys

    ap = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(ap):
        return
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", ap], check=False)
        elif os.name == "nt":
            os.startfile(ap)  # type: ignore[attr-defined]
        elif shutil.which("xdg-open"):
            subprocess.run(["xdg-open", ap], check=False)
    except OSError:
        pass


def _save_1d_histogram_mpl(path: str, data: np.ndarray, bins: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = np.asarray(data, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if vals.size:
        ax.hist(vals, bins=max(1, int(bins)), color="steelblue", edgecolor="black")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    _open_saved_plot_path(path)


def _save_1d_lineplot_mpl(
    path: str,
    data: np.ndarray,
    *,
    show_lines: bool,
    show_points: bool,
    show_smooth: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(data, dtype=np.float64).squeeze()
    n = int(y.shape[0])
    x = np.arange(n, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not (show_lines or show_points or show_smooth):
        ax.text(0.5, 0.5, "No display option selected", ha="center", va="center")
    else:
        if show_lines and show_points:
            ax.plot(x, y, "-o", color="C0", markersize=3)
        elif show_lines:
            ax.plot(x, y, "-", color="C0")
        elif show_points:
            ax.plot(x, y, "o", color="C0", linestyle="none", markersize=4)
        if show_smooth:
            sm = _cubic_smooth_xy(x, y)
            if sm is not None:
                xf, yf = sm
                ax.plot(xf, yf, "--", color="C1", label="cubic spline")
                ax.legend(loc="best")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    _open_saved_plot_path(path)


def _compound_field_float1d(rec: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(rec[name], dtype=np.float64).reshape(-1)


def _compound_xy_vectors(
    rec: np.ndarray, x_key: str, y_key: str
) -> tuple[np.ndarray, np.ndarray]:
    """Row index or field for X; field for Y."""
    yv = _compound_field_float1d(rec, y_key)
    if x_key == COMPOUND_PLOT_INDEX:
        xv = np.arange(int(rec.shape[0]), dtype=np.float64)
    else:
        xv = _compound_field_float1d(rec, x_key)
    return xv, yv


def _mask_for_log_axes(
    x: np.ndarray, y: np.ndarray, log_x: bool, log_y: bool
) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    if log_x:
        m &= x > 0
    if log_y:
        m &= y > 0
    return x[m], y[m]


def _plotext_compound_line(
    plot_plt,
    rec: np.ndarray,
    *,
    x_key: str,
    y_key: str,
    log_x: bool,
    log_y: bool,
) -> None:
    xv, yv = _compound_xy_vectors(rec, x_key, y_key)
    xv, yv = _mask_for_log_axes(xv, yv, log_x, log_y)
    plot_plt.clear_figure()
    plot_plt.xscale("log" if log_x else "linear")
    plot_plt.yscale("log" if log_y else "linear")
    xlab = "Index" if x_key == COMPOUND_PLOT_INDEX else str(x_key)
    plot_plt.title(f"{y_key} vs {xlab}")
    plot_plt.xlabel(xlab)
    plot_plt.ylabel(str(y_key))
    if xv.size == 0:
        return
    plot_plt.plot(
        xv,
        yv,
        color="cyan",
        marker="braille",
    )


def _save_compound_lineplot_mpl(
    path: str,
    rec: np.ndarray,
    *,
    x_key: str,
    y_key: str,
    log_x: bool,
    log_y: bool,
    show_lines: bool,
    show_points: bool,
    show_smooth: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xv, yv = _compound_xy_vectors(rec, x_key, y_key)
    xv, yv = _mask_for_log_axes(xv, yv, log_x, log_y)
    xlab = "Index" if x_key == COMPOUND_PLOT_INDEX else str(x_key)
    ylab = str(y_key)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if xv.size == 0:
        ax.text(0.5, 0.5, "No finite points to plot", ha="center", va="center")
    elif not (show_lines or show_points or show_smooth):
        ax.text(0.5, 0.5, "No display option selected", ha="center", va="center")
    else:
        if show_lines and show_points:
            ax.plot(xv, yv, "-o", color="C0", markersize=3)
        elif show_lines:
            ax.plot(xv, yv, "-", color="C0")
        elif show_points:
            ax.plot(xv, yv, "o", color="C0", linestyle="none", markersize=4)
        if show_smooth:
            sm = _cubic_smooth_xy(xv, yv)
            if sm is not None:
                xf, yf = sm
                ax.plot(xf, yf, "--", color="C1", label="cubic spline")
                ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    _open_saved_plot_path(path)


def _save_compound_histogram_mpl(
    path: str, rec: np.ndarray, field: str, bins: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = _compound_field_float1d(rec, field)
    vals = vals[np.isfinite(vals)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if vals.size:
        ax.hist(vals, bins=max(1, int(bins)), color="steelblue", edgecolor="black")
    ax.set_xlabel(str(field))
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    _open_saved_plot_path(path)


class SavePlotScreen(ModalScreen):
    """Save 1D dataset as PDF: histogram or line plot (matplotlib).

    Compound 1D datasets: line plot with X/Y field choosers and optional log axes;
    histogram uses one numeric field.

    After a successful save, the file is opened with the OS default app (e.g. Preview).
    """

    BINDINGS = [
        Binding("escape,q", "close_save", "Close", show=True, priority=True),
        Binding("Q", "quit_application", "Quit app", show=False, priority=True),
    ]

    def __init__(
        self,
        data: np.ndarray,
        dset_name: str,
        *,
        compound_x: str | None = None,
        compound_y: str | None = None,
        compound_log_x: bool = False,
        compound_log_y: bool = False,
        plot_settings_only: bool = False,
        histogram_settings_only: bool = False,
        id=None,
    ) -> None:
        super().__init__(id=id)
        self._data = np.asarray(data)
        self._dset_name = dset_name
        self._is_compound = self._data.dtype.fields is not None and self._data.ndim == 1
        self._histogram_settings_only = (
            bool(histogram_settings_only) and self._is_compound
        )
        self._plot_settings_only = (
            bool(plot_settings_only)
            and self._is_compound
            and not self._histogram_settings_only
        )
        self._numeric_names = (
            dtype_numeric_field_names(self._data.dtype) if self._is_compound else []
        )
        if self._is_compound and self._numeric_names:
            self._default_bins = _default_histogram_bins(
                np.asarray(self._data[self._numeric_names[0]], dtype=np.float64)
            )
        else:
            self._default_bins = _default_histogram_bins(self._data)
        safe = dset_name.replace("/", "_").strip() or "plot"
        self._default_filename = f"{safe}.pdf"
        self._init_compound_x = compound_x
        self._init_compound_y = compound_y
        self._init_compound_log_x = compound_log_x
        self._init_compound_log_y = compound_log_y

    def _compound_select_params(
        self,
    ) -> tuple[
        list[tuple[str, str]],
        list[tuple[str, str]],
        list[tuple[str, str]],
        str,
        str,
        str,
    ]:
        x_opts = [("Index", COMPOUND_PLOT_INDEX)] + [
            (n, n) for n in self._numeric_names
        ]
        y_opts = [(n, n) for n in self._numeric_names]
        h_opts = y_opts
        dx = self._init_compound_x
        dy = self._init_compound_y
        if dx is None or not any(dx == v for _, v in x_opts):
            dx = COMPOUND_PLOT_INDEX
        if dy is None or not any(dy == v for _, v in y_opts):
            dy = self._numeric_names[0] if self._numeric_names else ""
        hf = (
            dy
            if dy in self._numeric_names
            else (self._numeric_names[0] if self._numeric_names else "")
        )
        return x_opts, y_opts, h_opts, dx, dy, hf

    def _yield_compound_plot_field_widgets(self) -> ComposeResult:
        x_opts, y_opts, h_opts, dx, dy, hf = self._compound_select_params()
        with VerticalGroup(classes="compound-xy-opts"):
            with HorizontalGroup(classes="compound-axis-row"):
                yield Select(x_opts, id="compound_x", value=dx)
                yield Checkbox(
                    "Log X",
                    value=bool(self._init_compound_log_x),
                    id="compound_log_x",
                )
            with HorizontalGroup(classes="compound-axis-row"):
                yield Select(y_opts, id="compound_y", value=dy)
                yield Checkbox(
                    "Log Y",
                    value=bool(self._init_compound_log_y),
                    id="compound_log_y",
                )
        with VerticalGroup(classes="compound-hist-opts"):
            yield Select(h_opts, id="compound_hist_field", value=hf)

    def compose(self) -> ComposeResult:
        if self._histogram_settings_only:
            _, _, h_opts, _, _, hf = self._compound_select_params()
            dlg_classes = "mode-histogram compound-dataset histogram-settings-only"
            with Vertical(id="save-plot-dialog", classes=dlg_classes):
                with VerticalScroll(id="save_plot_scroll"):
                    with VerticalGroup():
                        yield Static(
                            f"Histogram · {self._dset_name}", id="save_plot_title"
                        )
                        with VerticalGroup(classes="compound-hist-opts"):
                            yield Select(h_opts, id="compound_hist_field", value=hf)
                            yield Button(
                                "Plot",
                                variant="primary",
                                id="confirm_hist_plot_btn",
                            )
            yield Footer()
            return

        if self._plot_settings_only:
            dlg_classes = "mode-lineplot compound-dataset plot-settings-only"
            with Vertical(id="save-plot-dialog", classes=dlg_classes):
                with VerticalScroll(id="save_plot_scroll"):
                    with VerticalGroup():
                        yield Static(f"Plot · {self._dset_name}", id="save_plot_title")
                        yield from self._yield_compound_plot_field_widgets()
                        yield Button("Plot", variant="primary", id="confirm_plot_btn")
            yield Footer()
            return

        dlg_classes = "mode-lineplot"
        if self._is_compound:
            dlg_classes += " compound-dataset"
        with Vertical(id="save-plot-dialog", classes=dlg_classes):
            with VerticalScroll(id="save_plot_scroll"):
                with VerticalGroup():
                    yield Static(f"Save plot · {self._dset_name}", id="save_plot_title")
                    yield Select(
                        (
                            ("Line plot", "lineplot"),
                            ("Histogram", "histogram"),
                        ),
                        id="plot_save_mode",
                        value="lineplot",
                        allow_blank=False,
                    )
                    if self._is_compound:
                        yield from self._yield_compound_plot_field_widgets()
                    yield Input(
                        value=str(self._default_bins),
                        id="save_bins",
                        placeholder="Bins",
                        type="integer",
                        classes="hist-opts",
                    )
                    with HorizontalGroup(classes="line-opts"):
                        yield Checkbox("Lines", value=True, id="line_lines")
                        yield Checkbox("Points", value=False, id="line_points")
                        yield Checkbox(
                            "Smooth (cubic spline)", value=False, id="line_smooth"
                        )
                    yield Input(value=self._default_filename, id="save_filename")
                    yield Button("Save", variant="primary", id="save_plot_btn")
        yield Footer()

    def _apply_select_border_titles(self) -> None:
        for sel_id, title in (
            ("#plot_save_mode", "Plot type"),
            ("#compound_x", "X axis"),
            ("#compound_y", "Y axis"),
            ("#compound_hist_field", "Field"),
        ):
            try:
                self.query_one(sel_id, Select).border_title = title
            except NoMatches:
                pass

    def on_mount(self) -> None:
        self._set_mode_class()
        self._apply_select_border_titles()
        if self._histogram_settings_only:
            self.query_one("#compound_hist_field", Select).focus()
        elif self._plot_settings_only:
            self.query_one("#compound_x", Select).focus()
        else:
            self.query_one("#save_bins", Input).border_title = "Bins"
            self.query_one("#save_filename", Input).border_title = "Filename"
            self.query_one("#save_filename", Input).focus()

    @staticmethod
    def _mode_str(value: object) -> str:
        """Normalize Select value to ``lineplot`` / ``histogram``.

        Do not use ``Select.NULL`` / ``Select.BLANK``: names differ across Textual versions.
        Our options are only these two strings; anything else is treated as line plot.
        """
        if value == "histogram":
            return "histogram"
        return "lineplot"

    def _mode(self) -> str:
        if getattr(self, "_histogram_settings_only", False):
            return "histogram"
        if getattr(self, "_plot_settings_only", False):
            return "lineplot"
        sel = self.query_one("#plot_save_mode", Select)
        return self._mode_str(sel.value)

    def _set_mode_class(self, mode: str | None = None) -> None:
        dlg = self.query_one("#save-plot-dialog", Vertical)
        if mode is None:
            mode = self._mode()
        parts = [f"mode-{mode}"]
        if getattr(self, "_is_compound", False):
            parts.append("compound-dataset")
        if getattr(self, "_plot_settings_only", False):
            parts.append("plot-settings-only")
        if getattr(self, "_histogram_settings_only", False):
            parts.append("histogram-settings-only")
        dlg.set_classes(" ".join(parts))

    @on(Select.Changed, "#plot_save_mode")
    def _on_mode_changed(self, event: Select.Changed) -> None:
        # Use the message value: ``Select.value`` can still be the previous value
        # during this handler on some Textual versions / focus paths.
        mode = self._mode_str(event.value)
        self._set_mode_class(mode)

    def action_close_save(self) -> None:
        if getattr(self, "_histogram_settings_only", False):
            fn = getattr(self.app, "_on_compound_histogram_browse_cancelled", None)
            if callable(fn):
                fn()
        elif getattr(self, "_plot_settings_only", False):
            fn = getattr(self.app, "_on_compound_plot_settings_cancelled", None)
            if callable(fn):
                fn()
        self.app.pop_screen()

    def action_quit_application(self) -> None:
        self.app.exit()

    @on(Button.Pressed, "#confirm_plot_btn")
    def _on_confirm_plot_pressed(self) -> None:
        self._persist_compound_plot_settings()
        self.app.pop_screen()
        fn = getattr(self.app, "_on_compound_plot_settings_confirmed", None)
        if callable(fn):
            fn()

    @on(Button.Pressed, "#confirm_hist_plot_btn")
    def _on_confirm_hist_plot_pressed(self) -> None:
        hf = self.query_one("#compound_hist_field", Select).value
        if not hf:
            self.app.notify("Select a histogram field", severity="warning", timeout=2)
            return
        self.app._compound_browse_histogram_field = str(hf)
        self.app.pop_screen()
        fn = getattr(self.app, "_on_compound_histogram_browse_confirmed", None)
        if callable(fn):
            fn()

    def _persist_compound_plot_settings(self) -> None:
        if not self._is_compound:
            return
        app = self.app
        app._compound_plot_x = str(self.query_one("#compound_x", Select).value)
        app._compound_plot_y = str(self.query_one("#compound_y", Select).value)
        app._compound_plot_log_x = bool(
            self.query_one("#compound_log_x", Checkbox).value
        )
        app._compound_plot_log_y = bool(
            self.query_one("#compound_log_y", Checkbox).value
        )

    @on(Button.Pressed, "#save_plot_btn")
    def _on_save_pressed(self) -> None:
        path = self.query_one("#save_filename", Input).value.strip()
        if not path:
            self.app.notify("Enter a filename", severity="warning", timeout=2)
            return
        mode = self._mode()
        try:
            if self._is_compound:
                self._persist_compound_plot_settings()
                if mode == "histogram":
                    try:
                        bins = int(self.query_one("#save_bins", Input).value)
                    except ValueError:
                        self.app.notify(
                            "Invalid bin count", severity="warning", timeout=2
                        )
                        return
                    bins = max(1, bins)
                    hf = self.query_one("#compound_hist_field", Select).value
                    if not hf:
                        self.app.notify(
                            "Select a histogram field", severity="warning", timeout=2
                        )
                        return
                    _save_compound_histogram_mpl(path, self._data, str(hf), bins)
                else:
                    lines = self.query_one("#line_lines", Checkbox).value
                    points = self.query_one("#line_points", Checkbox).value
                    smooth = self.query_one("#line_smooth", Checkbox).value
                    if not (lines or points or smooth):
                        self.app.notify(
                            "Select at least one of lines, points, or smooth",
                            severity="warning",
                            timeout=3,
                        )
                        return
                    xk = str(self.query_one("#compound_x", Select).value)
                    yk = str(self.query_one("#compound_y", Select).value)
                    _save_compound_lineplot_mpl(
                        path,
                        self._data,
                        x_key=xk,
                        y_key=yk,
                        log_x=self.query_one("#compound_log_x", Checkbox).value,
                        log_y=self.query_one("#compound_log_y", Checkbox).value,
                        show_lines=lines,
                        show_points=points,
                        show_smooth=smooth,
                    )
            elif mode == "histogram":
                try:
                    bins = int(self.query_one("#save_bins", Input).value)
                except ValueError:
                    self.app.notify("Invalid bin count", severity="warning", timeout=2)
                    return
                bins = max(1, bins)
                _save_1d_histogram_mpl(path, self._data, bins)
            else:
                lines = self.query_one("#line_lines", Checkbox).value
                points = self.query_one("#line_points", Checkbox).value
                smooth = self.query_one("#line_smooth", Checkbox).value
                if not (lines or points or smooth):
                    self.app.notify(
                        "Select at least one of lines, points, or smooth",
                        severity="warning",
                        timeout=3,
                    )
                    return
                _save_1d_lineplot_mpl(
                    path,
                    self._data,
                    show_lines=lines,
                    show_points=points,
                    show_smooth=smooth,
                )
            self.app.notify(f"Saved to {path}", timeout=3)
            self.app.pop_screen()
        except OSError as e:
            self.app.notify(f"Could not save: {e}", severity="error", timeout=8)


def _format_stat_index(arr: np.ndarray, flat_index: int) -> str:
    """Human-readable multi-index for ``flat_index`` into ``arr`` (cf. ``argmax`` / ``argmin``)."""
    t = np.unravel_index(int(flat_index), arr.shape)
    if len(t) == 1:
        return str(int(t[0]))
    return "(" + ", ".join(str(int(x)) for x in t) + ")"


class DatasetSummaryScreen(ModalScreen):
    """Modal stat cards: border title + value ``Static`` (Textual defaults for layout)."""

    BINDINGS = [
        Binding(
            "left,h,escape,q",
            "quit_summary",
            "Close",
            show=True,
            priority=True,
        ),
        Binding("Q", "quit_application", "Quit app", show=True, priority=True),
        Binding("J", "scroll_summary_down", "Scroll down", priority=True),
        Binding("K", "scroll_summary_up", "Scroll up", priority=True),
        Binding("u", "scroll_summary_page_up", "Page up", priority=True),
        Binding("d", "scroll_summary_page_down", "Page down", priority=True),
    ]

    def __init__(
        self,
        data: np.ndarray | None = None,
        *,
        streaming_stats: dict[str, float] | None = None,
        id=None,
    ) -> None:
        super().__init__(id=id)
        if streaming_stats is not None:
            st = streaming_stats
            self._stat_rows = [
                [
                    ("Mean", f"{float(st['mean']):.5g}"),
                    ("Var", f"{float(st['var']):.5g}"),
                    ("Std", f"{float(st['std']):.5g}"),
                    ("L2 norm", f"{float(st['L2 norm']):.5g}"),
                ],
                [
                    ("Max", f"{float(st['max']):.5g}"),
                    ("Max idx", "— (chunked scan)"),
                    ("Min", f"{float(st['min']):.5g}"),
                    ("Min idx", "— (chunked scan)"),
                ],
            ]
            return
        if data is None:
            raise ValueError("DatasetSummaryScreen requires data or streaming_stats")
        arr = np.asarray(data)
        flat_max = int(np.argmax(arr))
        flat_min = int(np.argmin(arr))
        # Row 1: moments / norms; row 2: extrema and indices (same ddof as ``np.std``).
        self._stat_rows: list[list[tuple[str, str]]] = [
            [
                ("Mean", f"{float(np.mean(arr)):.5g}"),
                ("Var", f"{float(np.var(arr)):.5g}"),
                ("Std", f"{float(np.std(arr)):.5g}"),
                ("L2 norm", f"{float(np.linalg.norm(arr)):.5g}"),
            ],
            [
                ("Max", f"{float(np.max(arr)):.5g}"),
                ("Max idx", _format_stat_index(arr, flat_max)),
                ("Min", f"{float(np.min(arr)):.5g}"),
                ("Min idx", _format_stat_index(arr, flat_min)),
            ],
        ]

    def compose(self) -> ComposeResult:
        with Vertical(id="summary_dialog"):
            yield Static("Summary", id="summary_modal_title")
            with VerticalScroll(id="summary_stat_scroll"):
                with VerticalGroup(id="summary_stat_grid"):
                    for row in self._stat_rows:
                        # ``Horizontal`` / ``Vertical`` default to ``height: 1fr`` and stretch
                        # rows to fill the scroll viewport; ``*Group`` uses ``height: auto``.
                        with HorizontalGroup(classes="summary-stat-row"):
                            for title, value in row:
                                box = VerticalGroup(
                                    Static(
                                        value,
                                        classes="summary-stat-value",
                                        markup=False,
                                    ),
                                    classes="summary-stat-box",
                                )
                                box.border_title = title
                                yield box
        yield Footer()

    def action_quit_summary(self) -> None:
        self.app.pop_screen()

    def action_quit_application(self) -> None:
        self.app.exit()

    def _summary_scroll(self) -> VerticalScroll:
        return self.query_one("#summary_stat_scroll", VerticalScroll)

    def action_scroll_summary_down(self) -> None:
        self._summary_scroll().scroll_down()

    def action_scroll_summary_up(self) -> None:
        self._summary_scroll().scroll_up()

    def action_scroll_summary_page_down(self) -> None:
        self._summary_scroll().scroll_page_down()

    def action_scroll_summary_page_up(self) -> None:
        self._summary_scroll().scroll_page_up()


class JumpToLineScreen(ModalScreen):
    """Enter a row index (0-based, matching table row labels) for plain 1D/2D tables."""

    BINDINGS = [
        Binding("escape,q", "close_jump", "Close", show=True, priority=True),
        Binding("Q", "quit_application", "Quit app", show=False, priority=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="jump-line-dialog"):
            yield Input(
                placeholder="Row index (0-based)",
                type="integer",
                id="jump_line_input",
            )
        yield Footer()

    def on_mount(self) -> None:
        dlg = self.query_one("#jump-line-dialog", Vertical)
        dlg.border_title = "Jump to line"
        self.query_one("#jump_line_input", Input).focus()

    def action_close_jump(self) -> None:
        self.app.pop_screen()

    def action_quit_application(self) -> None:
        self.app.exit()

    @on(Input.Submitted, "#jump_line_input")
    def _on_submitted(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        if not raw:
            self.app.pop_screen()
            return
        try:
            idx = int(raw)
        except ValueError:
            self.app.notify("Enter a valid integer", severity="warning", timeout=2)
            return
        if idx < 0:
            self.app.notify(
                "Row index must be non-negative", severity="warning", timeout=2
            )
            return
        self.app.apply_jump_to_line_row(idx)
        self.app.pop_screen()


class GroupFindScreen(ModalScreen):
    """Live filter of names in the current HDF5 group (``/``), Yazi-style."""

    BINDINGS = [
        Binding("escape,q", "cancel_find", "Cancel", show=True, priority=True),
        Binding("Q", "quit_application", "Quit app", show=False, priority=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="group-find-dialog"):
            yield Input(placeholder="", id="group_find_input")

    def on_mount(self) -> None:
        dlg = self.query_one("#group-find-dialog", Vertical)
        dlg.border_title = "Find in group:"
        inp = self.query_one("#group_find_input", Input)
        inp.focus()
        self.app.apply_group_find_query("")

    @on(Input.Changed, "#group_find_input")
    def _on_input_changed(self, event: Input.Changed) -> None:
        self.app.apply_group_find_query(event.value)

    @on(Input.Submitted, "#group_find_input")
    def _on_submitted(self, event: Input.Submitted) -> None:
        self.app.finish_group_find(cancelled=False)
        self.app.pop_screen()

    def action_cancel_find(self) -> None:
        self.app.finish_group_find(cancelled=True)
        self.app.pop_screen()

    def action_quit_application(self) -> None:
        self.app.exit()


_MODAL_SCREENS_ALL = (
    AttributeScreen,
    SavePlotScreen,
    DatasetSummaryScreen,
    JumpToLineScreen,
    GroupFindScreen,
)
_MODAL_SCREENS_CHILD = (
    SavePlotScreen,
    DatasetSummaryScreen,
    JumpToLineScreen,
    GroupFindScreen,
)

_KEYS_HELP_TABLE_1D = "keys: j/k row; J/K jump ±10; y copy; esc quit"
_KEYS_HELP_TABLE_2D = "keys: h/j/k/l; H/J/K/L jump ±10; y copy; esc quit"


def _reprint_lazy_datatable(
    preview, refresh, message: str, *, timeout: float = 5.0
) -> None:
    """Run lazy refresh; on success notify and focus the data table."""
    try:
        refresh()
    except Exception as e:
        preview.notify(f"Could not read dataset: {e}", severity="error")
        return
    preview.notify(message, timeout=timeout)
    preview._df.focus()


def _cell_value_to_str(value: object) -> str:
    """Plain string for DataTable cell contents (clipboard / display)."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Text):
        return value.plain
    return str(value)


def _scalar_clipboard_str(value: object) -> str:
    """Clipboard text from a raw scalar (full float repr, not table formatting)."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Text):
        return value.plain
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            return np.array2string(value, max_line_width=10_000)
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, (complex, np.complexfloating)):
        return repr(complex(value))
    return str(value)


def _dt_row_label_plain(dt: DataTable, row_index: int) -> str:
    rk = dt.ordered_rows[row_index]
    lab = dt.rows[rk].label
    return lab.plain if lab is not None else ""


def _dt_col_label_plain(dt: DataTable, col_index: int) -> str:
    ck = dt.ordered_columns[col_index]
    return dt.columns[ck].label.plain


def _datatable_source_clipboard_str(app, dt: DataTable) -> str | None:
    """HDF5 / ``_data`` scalar for the focused cell, or None to use displayed cell text."""
    if dt.row_count == 0 or not dt.columns:
        return None
    r, c = dt.cursor_coordinate
    if r < 0 or c < 0:
        return None

    if getattr(app, "_lazy_1d_windowed", False) and getattr(
        app, "_lazy_dset_path", None
    ):
        try:
            dset = app._file[app._lazy_dset_path]
            idx = int(app._lazy_1d_window_start) + r
            if getattr(app, "_lazy_structured_1d", False):
                names = dset.dtype.names
                if not names:
                    return None
                nf = len(names)
                fv = min(LAZY_STRUCT_FIELD_WINDOW, nf)
                mx = max(0, nf - fv)
                fs = max(
                    0,
                    min(int(getattr(app, "_lazy_struct_field_start", 0)), mx),
                )
                gf = fs + c
                if gf < 0 or gf >= nf:
                    return None
                fname = names[gf]
                return _scalar_clipboard_str(np.asarray(dset[fname][idx]).item())
            return _scalar_clipboard_str(np.asarray(dset[idx]).item())
        except Exception:
            return None
    if getattr(app, "_lazy_2d_windowed", False) and getattr(
        app, "_lazy_dset_path", None
    ):
        try:
            dset = app._file[app._lazy_dset_path]
            gr = int(app._lazy_2d_window_row) + r
            gc = int(app._lazy_2d_window_col) + c
            return _scalar_clipboard_str(np.asarray(dset[gr, gc]).item())
        except Exception:
            return None

    data = getattr(app, "_data", None)
    if (
        app.has_class("view-dataset")
        and data is not None
        and not getattr(app, "_lazy_mode", False)
    ):
        if is_dataframe(data):
            names = get_colnames(data)
            if not names or c >= len(names):
                return None
            return _scalar_clipboard_str(data[r][names[c]])
        arr = np.asarray(data)
        if arr.ndim == 1 and len(dt.ordered_columns) == 1:
            return _scalar_clipboard_str(arr[r])
        if arr.ndim == 2:
            return _scalar_clipboard_str(arr[r, c])
        return None

    preview = getattr(app, "_preview", None)
    if preview is not None and preview.has_class("browse-preview-table"):
        name = app.get_selected_itemname()
        if not name:
            return None
        path = h5_child_path(app._cur_dir, name)
        try:
            dset = app._file[path]
        except Exception:
            return None
        rl = _dt_row_label_plain(dt, r)
        cl = _dt_col_label_plain(dt, c)
        if rl == "…" or cl == "…":
            return None
        try:
            gr = int(rl)
        except ValueError:
            return None
        try:
            if dset.dtype.fields is not None and dset.ndim == 1:
                names = dset.dtype.names
                if not names or cl not in names:
                    return None
                return _scalar_clipboard_str(np.asarray(dset[cl][gr]).item())
            if dset.ndim == 1:
                return _scalar_clipboard_str(np.asarray(dset[gr]).item())
            if dset.ndim == 2:
                gc = int(cl)
                return _scalar_clipboard_str(np.asarray(dset[gr, gc]).item())
        except Exception:
            return None
        return None

    return None


class MyDataTable(DataTable):
    """Jump distance for H/J/K/L (capital) — move cursor by this many rows/columns."""

    _JUMP = 10

    BINDINGS = [
        Binding("enter", "select_cursor", "Select", show=False),
        Binding("up,k", "cursor_up", "Cursor up", show=False),
        Binding("down,j", "cursor_down", "Cursor down", show=False),
        # Lowercase h/l for columns; priority so app goto_parent/goto_child does not steal keys.
        Binding("h", "cursor_left", "Left", show=False, priority=True),
        Binding("l", "cursor_right", "Right", show=False, priority=True),
        Binding("left", "cursor_left", "Cursor left", show=False),
        Binding("right", "cursor_right", "Cursor right", show=False),
        Binding("H", "jump_left", show=False, priority=True),
        Binding("J", "jump_down", show=False, priority=True),
        Binding("K", "jump_up", show=False, priority=True),
        Binding("L", "jump_right", show=False, priority=True),
        Binding("pageup,u", "page_up", "Page up", show=False),
        Binding("pagedown,d", "page_down", "Page down", show=False),
        Binding("g", "scroll_top", "Top", show=False),
        Binding("G", "scroll_bottom", "Bottom", show=False),
        Binding("y", "copy_cell", "Copy", show=True),
    ]

    def __init__(self, id):
        super().__init__(id=id)

    def action_jump_up(self) -> None:
        self._set_hover_cursor(False)
        if self.row_count == 0 or not self.columns:
            return
        ct = self.cursor_type
        if not self.show_cursor or ct == "none":
            super().action_scroll_up()
            return
        r, c = self.cursor_coordinate
        if ct in ("cell", "row"):
            self.move_cursor(row=max(0, r - self._JUMP), scroll=True)
        elif ct == "column":
            # Native ``j`` / ``k`` only scroll for column cursor; keep J/K as vertical jump.
            self.move_cursor(row=max(0, r - self._JUMP), column=c, scroll=True)
        else:
            super().action_scroll_up()

    def action_jump_down(self) -> None:
        self._set_hover_cursor(False)
        if self.row_count == 0 or not self.columns:
            return
        ct = self.cursor_type
        if not self.show_cursor or ct == "none":
            super().action_scroll_down()
            return
        r, c = self.cursor_coordinate
        if ct in ("cell", "row"):
            self.move_cursor(row=min(self.row_count - 1, r + self._JUMP), scroll=True)
        elif ct == "column":
            self.move_cursor(
                row=min(self.row_count - 1, r + self._JUMP), column=c, scroll=True
            )
        else:
            super().action_scroll_down()

    def action_jump_left(self) -> None:
        self._set_hover_cursor(False)
        if self.row_count == 0 or not self.columns:
            return
        ct = self.cursor_type
        if self.show_cursor and ct in ("cell", "column"):
            _, c = self.cursor_coordinate
            self.move_cursor(column=max(0, c - self._JUMP), scroll=True)
        else:
            super().action_scroll_left()

    def action_jump_right(self) -> None:
        self._set_hover_cursor(False)
        if self.row_count == 0 or not self.columns:
            return
        ct = self.cursor_type
        if self.show_cursor and ct in ("cell", "column"):
            _, c = self.cursor_coordinate
            self.move_cursor(
                column=min(len(self.columns) - 1, c + self._JUMP), scroll=True
            )
        else:
            super().action_scroll_right()

    def action_copy_cell(self) -> None:
        """Copy the focused cell value to the app clipboard."""
        if self.row_count == 0:
            return
        s = _datatable_source_clipboard_str(self.app, self)
        if s is None:
            try:
                s = _cell_value_to_str(self.get_cell_at(self.cursor_coordinate))
            except Exception:
                return
        self.app.copy_to_clipboard(s)
        self.app.notify("Copied to clipboard", timeout=1.5)

    def _compute_row_renderables(self, row_index: int):
        """Center column titles in the header row (Textual left-aligns labels by default)."""
        if row_index != -1:
            return super()._compute_row_renderables(row_index)
        from rich.align import Align
        from textual.widgets._data_table import RowRenderables

        pad = int(self.cell_padding)
        header_row = []
        for column in self.ordered_columns:
            w = column.get_render_width(self)
            inner = max(1, w - 2 * pad)
            header_row.append(Align(column.label, "center", width=inner))
        return RowRenderables(None, header_row)

    def update(self, value):
        self.clear(columns=True)
        self.add_columns(*get_colnames(value))
        for i, row in enumerate(value):
            # Match browse/lazy tables: pre-format so sign alignment isn't column-dependent
            # (Textual left-aligns the first column vs others when given raw scalars).
            cells = tuple(_format_table_cell(e) for e in row.item())
            self.add_row(*cells, label=str(i))

    def update_1d(self, arr: np.ndarray) -> None:
        """Fill one unnamed column; row labels are ``0``, ``1``, … (no colon; text preview uses ``i:``)."""
        a = np.asarray(arr)
        self.clear(columns=True)
        self.add_columns("")
        if a.size == 0:
            return
        for i in range(int(a.shape[0])):
            self.add_row(
                _format_table_cell(a[i]),
                label=str(i),
            )

    def update_2d(self, arr: np.ndarray) -> None:
        """Fill one column per matrix column; row index is a non-interactive label."""
        a = np.asarray(arr)
        self.clear(columns=True)
        if a.size == 0:
            return
        nrows, ncols = int(a.shape[0]), int(a.shape[1])
        self.add_columns(*(str(j) for j in range(ncols)))
        for i in range(nrows):
            row = a[i]
            cells = tuple(_format_table_cell(row[j]) for j in range(ncols))
            self.add_row(*cells, label=str(i))

    def update_1d_window(self, arr: np.ndarray, base_index: int) -> None:
        """1D table with global row labels ``base_index`` … (sliding window)."""
        a = np.asarray(arr)
        self.clear(columns=True)
        self.add_columns("")
        if a.size == 0:
            return
        bi = int(base_index)
        for i in range(int(a.shape[0])):
            self.add_row(
                _format_table_cell(a[i]),
                label=str(bi + i),
            )

    def update_structured_1d_window(
        self,
        chunk: np.ndarray,
        base_index: int,
        field_names: tuple[str, ...],
        *,
        field_start: int,
        n_fields_total: int,
    ) -> None:
        """Sliding window over 1D compound rows; ``field_names`` is a slice of HDF5 fields."""
        self.clear(columns=True)
        col_labels = [str(f) for f in field_names]
        self.add_columns(*col_labels)
        if chunk.size == 0:
            self.border_title = ""
            return
        bi = int(base_index)
        lo = field_start
        hi = field_start + len(field_names) - 1
        if n_fields_total <= len(field_names):
            self.border_title = f"fields [0:{hi}] ({n_fields_total})"
        else:
            self.border_title = f"fields [{lo}:{hi}] / {n_fields_total}"
        for i in range(int(chunk.shape[0])):
            cells = tuple(_format_table_cell(chunk[f][i]) for f in field_names)
            self.add_row(*cells, label=str(bi + i))

    def update_2d_window(self, arr: np.ndarray, base_row: int, base_col: int) -> None:
        """2D grid with global column headers and row labels (sliding view)."""
        a = np.asarray(arr)
        self.clear(columns=True)
        if a.size == 0:
            return
        nrows, ncols = int(a.shape[0]), int(a.shape[1])
        br, bc = int(base_row), int(base_col)
        self.add_columns(*(str(bc + j) for j in range(ncols)))
        for i in range(nrows):
            row = a[i]
            cells = tuple(_format_table_cell(row[j]) for j in range(ncols))
            self.add_row(*cells, label=str(br + i))


class LazyH5DataTable(MyDataTable):
    """DataTable that loads visible ranges of large 1D/2D datasets as you navigate."""

    def _lazy_mode(self) -> str | None:
        app = self.app
        if getattr(app, "_lazy_2d_windowed", False):
            return "2d"
        if getattr(app, "_lazy_1d_windowed", False):
            return "1d"
        return None

    def action_cursor_down(self) -> None:
        la = self.app
        mode = self._lazy_mode()
        if mode == "2d":
            r, c = self.cursor_coordinate
            br = la._lazy_2d_window_row
            nr = la._lazy_2d_total_rows
            wr = LAZY_2D_WINDOW_ROWS
            g = br + r
            if r == self.row_count - 1 and g < nr - 1:
                new_g = g + 1
                la._lazy_2d_window_row = min(br + 1, max(0, nr - wr))
                la.refresh_lazy_2d_datatable(
                    cursor_row=new_g - la._lazy_2d_window_row,
                    cursor_col=c,
                )
                return
        elif mode == "1d":
            r, _ = self.cursor_coordinate
            base = la._lazy_1d_window_start
            n = la._lazy_1d_total
            w = LAZY_DATATABLE_WINDOW_SIZE
            g = base + r
            if r == self.row_count - 1 and g < n - 1:
                new_g = g + 1
                la._lazy_1d_window_start = min(base + 1, max(0, n - w))
                la.refresh_lazy_1d_datatable(
                    cursor_row=new_g - la._lazy_1d_window_start
                )
                return
        super().action_cursor_down()

    def action_cursor_up(self) -> None:
        la = self.app
        mode = self._lazy_mode()
        if mode == "2d":
            r, c = self.cursor_coordinate
            br = la._lazy_2d_window_row
            gr = br + r
            if r == 0 and gr > 0:
                new_g = gr - 1
                la._lazy_2d_window_row = max(0, br - 1)
                la.refresh_lazy_2d_datatable(
                    cursor_row=new_g - la._lazy_2d_window_row,
                    cursor_col=c,
                )
                return
        elif mode == "1d":
            r, _ = self.cursor_coordinate
            base = la._lazy_1d_window_start
            g = base + r
            if r == 0 and g > 0:
                new_g = g - 1
                la._lazy_1d_window_start = max(0, base - 1)
                la.refresh_lazy_1d_datatable(
                    cursor_row=new_g - la._lazy_1d_window_start
                )
                return
        super().action_cursor_up()

    def action_cursor_right(self) -> None:
        la = self.app
        if self._lazy_mode() == "2d":
            r, c = self.cursor_coordinate
            bc = la._lazy_2d_window_col
            nc = la._lazy_2d_total_cols
            wc = LAZY_2D_WINDOW_COLS
            gc = bc + c
            if c == len(self.columns) - 1 and gc < nc - 1:
                new_gc = gc + 1
                la._lazy_2d_window_col = min(bc + 1, max(0, nc - wc))
                la.refresh_lazy_2d_datatable(
                    cursor_row=r,
                    cursor_col=new_gc - la._lazy_2d_window_col,
                )
                return
        if self._lazy_mode() == "1d" and getattr(la, "_lazy_structured_1d", False):
            r, c = self.cursor_coordinate
            try:
                dset = la._file[la._lazy_dset_path]
                names = dset.dtype.names or ()
            except Exception:
                return super().action_cursor_right()
            nf = len(names)
            fv = min(LAZY_STRUCT_FIELD_WINDOW, nf)
            if nf > fv:
                fs = int(getattr(la, "_lazy_struct_field_start", 0))
                mx_fs = max(0, nf - fv)
                if c == len(self.columns) - 1 and fs + fv < nf:
                    la._lazy_struct_field_start = min(fs + 1, mx_fs)
                    la.refresh_lazy_1d_datatable(
                        cursor_row=r,
                        cursor_col=fv - 1,
                    )
                    return
        super().action_cursor_right()

    def action_cursor_left(self) -> None:
        la = self.app
        if self._lazy_mode() == "2d":
            r, c = self.cursor_coordinate
            bc = la._lazy_2d_window_col
            gc = bc + c
            if c == 0 and gc > 0:
                new_gc = gc - 1
                la._lazy_2d_window_col = max(0, bc - 1)
                la.refresh_lazy_2d_datatable(
                    cursor_row=r,
                    cursor_col=new_gc - la._lazy_2d_window_col,
                )
                return
        if self._lazy_mode() == "1d" and getattr(la, "_lazy_structured_1d", False):
            r, c = self.cursor_coordinate
            try:
                dset = la._file[la._lazy_dset_path]
                names = dset.dtype.names or ()
            except Exception:
                return super().action_cursor_left()
            nf = len(names)
            fv = min(LAZY_STRUCT_FIELD_WINDOW, nf)
            fs = int(getattr(la, "_lazy_struct_field_start", 0))
            if nf > fv and c == 0 and fs > 0:
                la._lazy_struct_field_start = max(0, fs - 1)
                la.refresh_lazy_1d_datatable(cursor_row=r, cursor_col=0)
                return
        super().action_cursor_left()

    def action_jump_down(self) -> None:
        la = self.app
        mode = self._lazy_mode()
        if mode == "2d":
            r, c = self.cursor_coordinate
            br, bc = la._lazy_2d_window_row, la._lazy_2d_window_col
            gr, gc = br + r, bc + c
            ng = min(gr + self._JUMP, la._lazy_2d_total_rows - 1)
            la.lazy_2d_set_global_cell(ng, gc)
            return
        if mode == "1d":
            r, _ = self.cursor_coordinate
            g = la._lazy_1d_window_start + r
            ng = min(g + self._JUMP, la._lazy_1d_total - 1)
            la.lazy_1d_set_global_index(ng)
            return
        super().action_jump_down()

    def action_jump_up(self) -> None:
        la = self.app
        mode = self._lazy_mode()
        if mode == "2d":
            r, c = self.cursor_coordinate
            br, bc = la._lazy_2d_window_row, la._lazy_2d_window_col
            gr, gc = br + r, bc + c
            ng = max(0, gr - self._JUMP)
            la.lazy_2d_set_global_cell(ng, gc)
            return
        if mode == "1d":
            r, _ = self.cursor_coordinate
            g = la._lazy_1d_window_start + r
            ng = max(0, g - self._JUMP)
            la.lazy_1d_set_global_index(ng)
            return
        super().action_jump_up()

    def action_jump_right(self) -> None:
        la = self.app
        if self._lazy_mode() == "2d":
            r, c = self.cursor_coordinate
            br, bc = la._lazy_2d_window_row, la._lazy_2d_window_col
            gr, gc = br + r, bc + c
            ngc = min(gc + self._JUMP, la._lazy_2d_total_cols - 1)
            la.lazy_2d_set_global_cell(gr, ngc)
            return
        if self._lazy_mode() == "1d" and getattr(la, "_lazy_structured_1d", False):
            r, c = self.cursor_coordinate
            try:
                dset = la._file[la._lazy_dset_path]
                names = dset.dtype.names or ()
            except Exception:
                return super().action_jump_right()
            nf = len(names)
            fv = min(LAZY_STRUCT_FIELD_WINDOW, nf)
            if nf <= fv:
                return super().action_jump_right()
            mx_fs = max(0, nf - fv)
            fs = int(getattr(la, "_lazy_struct_field_start", 0))
            step = max(1, fv // 2)
            nfs = min(mx_fs, fs + step)
            la._lazy_struct_field_start = nfs
            new_c = min(c, fv - 1)
            la.refresh_lazy_1d_datatable(cursor_row=r, cursor_col=new_c)
            return
        super().action_jump_right()

    def action_jump_left(self) -> None:
        la = self.app
        if self._lazy_mode() == "2d":
            r, c = self.cursor_coordinate
            br, bc = la._lazy_2d_window_row, la._lazy_2d_window_col
            gr, gc = br + r, bc + c
            ngc = max(0, gc - self._JUMP)
            la.lazy_2d_set_global_cell(gr, ngc)
            return
        if self._lazy_mode() == "1d" and getattr(la, "_lazy_structured_1d", False):
            r, c = self.cursor_coordinate
            try:
                dset = la._file[la._lazy_dset_path]
                names = dset.dtype.names or ()
            except Exception:
                return super().action_jump_left()
            nf = len(names)
            fv = min(LAZY_STRUCT_FIELD_WINDOW, nf)
            if nf <= fv:
                return super().action_jump_left()
            fs = int(getattr(la, "_lazy_struct_field_start", 0))
            step = max(1, fv // 2)
            nfs = max(0, fs - step)
            la._lazy_struct_field_start = nfs
            new_c = min(c, fv - 1)
            la.refresh_lazy_1d_datatable(cursor_row=r, cursor_col=new_c)
            return
        super().action_jump_left()

    def action_scroll_top(self) -> None:
        la = self.app
        mode = self._lazy_mode()
        if mode == "2d":
            la.lazy_2d_set_global_cell(0, 0)
            return
        if mode == "1d":
            la.lazy_1d_set_global_index(0)
            return
        super().action_scroll_top()

    def action_scroll_bottom(self) -> None:
        la = self.app
        mode = self._lazy_mode()
        if mode == "2d":
            la.lazy_2d_set_global_cell(
                la._lazy_2d_total_rows - 1, la._lazy_2d_total_cols - 1
            )
            return
        if mode == "1d":
            la.lazy_1d_set_global_index(la._lazy_1d_total - 1)
            return
        super().action_scroll_bottom()

    def action_page_down(self) -> None:
        la = self.app
        mode = self._lazy_mode()
        if mode == "2d":
            r, c = self.cursor_coordinate
            br, bc = la._lazy_2d_window_row, la._lazy_2d_window_col
            gr, gc = br + r, bc + c
            step = max(1, LAZY_2D_WINDOW_ROWS // 2)
            ng = min(gr + step, la._lazy_2d_total_rows - 1)
            la.lazy_2d_set_global_cell(ng, gc)
            return
        if mode == "1d":
            r, _ = self.cursor_coordinate
            g = la._lazy_1d_window_start + r
            step = max(1, LAZY_DATATABLE_WINDOW_SIZE // 2)
            ng = min(g + step, la._lazy_1d_total - 1)
            la.lazy_1d_set_global_index(ng)
            return
        super().action_page_down()

    def action_page_up(self) -> None:
        la = self.app
        mode = self._lazy_mode()
        if mode == "2d":
            r, c = self.cursor_coordinate
            br, bc = la._lazy_2d_window_row, la._lazy_2d_window_col
            gr, gc = br + r, bc + c
            step = max(1, LAZY_2D_WINDOW_ROWS // 2)
            ng = max(0, gr - step)
            la.lazy_2d_set_global_cell(ng, gc)
            return
        if mode == "1d":
            r, _ = self.cursor_coordinate
            g = la._lazy_1d_window_start + r
            step = max(1, LAZY_DATATABLE_WINDOW_SIZE // 2)
            ng = max(0, g - step)
            la.lazy_1d_set_global_index(ng)
            return
        super().action_page_up()


def get_colnames(obj):
    """Return the column names of numpy recarray"""
    return obj.dtype.names


def is_dataframe(obj):
    """Checks if numpy array is a dataframe i.e., recarray"""
    return len(obj.dtype) != 0


def _is_numeric_dtype_for_plot(dt: np.dtype) -> bool:
    """Scalar numeric fields suitable for line/histogram plots (excludes nested / complex)."""
    if dt.fields is not None:
        return False
    return np.issubdtype(dt, np.number) and not np.issubdtype(dt, np.complexfloating)


def dtype_numeric_field_names(dtype: np.dtype) -> list[str]:
    """Names of top-level compound fields that are plottable scalars."""
    if dtype.fields is None or not dtype.names:
        return []
    return [
        str(n) for n in dtype.names if _is_numeric_dtype_for_plot(dtype.fields[n][0])
    ]


def has_compound_plot_fields(dtype: np.dtype) -> bool:
    return bool(dtype_numeric_field_names(dtype))


def is_plotable(obj):
    """1D/2D numeric plots; 1D compound with at least one numeric field; exclude ``object``."""
    squeezed = np.squeeze(np.asarray(obj))
    if squeezed.dtype == np.dtype("O"):
        return False
    if squeezed.dtype.fields is not None:
        return squeezed.ndim == 1 and has_compound_plot_fields(squeezed.dtype)
    return squeezed.ndim == 1 or squeezed.ndim == 2


def is_aggregatable(obj):
    return (
        isinstance(obj, np.ndarray)
        and np.issubdtype(obj.dtype, np.number)
        and obj.size > 1
    )


def is_histogram_eligible(obj) -> bool:
    """Plain numeric 1D data suitable for a value histogram (Plotext ``hist`` / ``bar``)."""
    if not is_plotable(obj):
        return False
    a = np.squeeze(np.asarray(obj))
    return a.ndim == 1 and np.issubdtype(a.dtype, np.number)


def _as_display_1d(arr: np.ndarray) -> np.ndarray:
    """Treat (N,1) / (1,N) numeric columns as 1D for text display."""
    a = np.asarray(arr)
    if a.ndim == 2 and a.size and min(a.shape) == 1 and a.dtype.fields is None:
        return a.ravel()
    return a


def is_plain_1d_array(arr: np.ndarray) -> bool:
    """Non-record 1D arrays: plain dtypes, fixed strings, or ``object`` (vlen strings)."""
    if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        return False
    return arr.dtype.fields is None


# Plain ndarray DataTable limits (text fallback beyond these).
MAX_TABLE_ELEMENTS = 200_000
MAX_TABLE_COLUMNS = 1024

# Thresholds for "large dataset" UX: plot/histogram show a toast (sampled terminal view);
# summary still uses double-confirm above _LARGE_AGGREGATE_ELEMENT_THRESHOLD.
_LARGE_PLOT_ELEMENT_THRESHOLD = 500_000
_LARGE_HISTOGRAM_ELEMENT_THRESHOLD = 3_000_000
_LARGE_AGGREGATE_ELEMENT_THRESHOLD = 10_000_000
# Double-press confirm window: ``App.notify`` timeout and ``set_timer`` (message pump).
_LARGE_CONFIRM_WINDOW_SEC = 4.0

# Above this element count, keep the ``h5py.Dataset`` and read slices (plain numeric or 1D vlen strings).
LAZY_THRESHOLD = MAX_TABLE_ELEMENTS
# Plot sampling for lazy datasets (terminal plot is downsampled).
_PLOT_MAX_1D = 20_000
_PLOT_MAX_2D = 200
# Summary streaming chunk size.
_AGG_CHUNK = 2_000_000
# Lazy DataTable window sizes (rows/cols loaded from HDF5 at a time).
LAZY_DATATABLE_WINDOW_SIZE = 300
LAZY_2D_WINDOW_ROWS = 120
LAZY_2D_WINDOW_COLS = 120
# 1D compound: show this many field columns at once; pan with h/l when there are more.
LAZY_STRUCT_FIELD_WINDOW = 48

# Browse-mode snapshot table (sidebar preview before pressing ``l``).
PREVIEW_TABLE_1D_ROWS = 25
PREVIEW_TABLE_2D_SIZE = 8
PREVIEW_TABLE_COMPOUND_ROWS = 12
PREVIEW_TABLE_COMPOUND_COLS = 8
# Full ``format_ndarray_text`` preview (any ndim); must stay small — whole array is read.
PREVIEW_BROWSE_TEXT_MAX_ELEMENTS = 512
_PREVIEW_ELLIPSIS = "…"


def _browse_preview_datatable_eligible(dset: h5py.Dataset) -> bool:
    """Datasets we show as a small browse DataTable: plain 1D/2D, object string 1D/2D, or 1D compound."""
    if dset.size == 0:
        return False
    if dset.dtype == np.dtype("O"):
        return dset.ndim in (1, 2)
    if dset.dtype.fields is not None:
        return (
            dset.ndim == 1
            and dset.dtype.names is not None
            and len(dset.dtype.names) > 0
        )
    return dset.ndim in (1, 2)


def fill_browse_preview_datatable(dset: h5py.Dataset, table: MyDataTable) -> None:
    """Fill browse preview: up to ``PREVIEW_TABLE_1D_ROWS`` / ``PREVIEW_TABLE_2D_SIZE`` with edge ``…``."""
    if dset.dtype.fields is not None and dset.ndim == 1:
        names = dset.dtype.names
        if not names:
            raise ValueError("compound browse preview requires named fields")
        n = int(dset.shape[0])
        m = min(PREVIEW_TABLE_COMPOUND_ROWS, n)
        chunk = np.asarray(dset[:m])
        nf = len(names)
        nshow = min(PREVIEW_TABLE_COMPOUND_COLS, nf)
        field_names = names[:nshow]
        more_cols = nf > nshow
        table.clear(columns=True)
        col_labels = [str(f) for f in field_names]
        if more_cols:
            col_labels.append(_PREVIEW_ELLIPSIS)
        table.add_columns(*col_labels)
        for i in range(m):
            cells = tuple(_format_table_cell(chunk[f][i]) for f in field_names)
            if more_cols:
                cells = cells + (_PREVIEW_ELLIPSIS,)
            table.add_row(*cells, label=str(i))
        if n > m:
            table.add_row(
                *((_PREVIEW_ELLIPSIS,) * len(col_labels)), label=_PREVIEW_ELLIPSIS
            )
        return

    if dset.ndim == 1:
        n = int(dset.shape[0])
        m = min(PREVIEW_TABLE_1D_ROWS, n)
        chunk = np.asarray(dset[:m])
        a = _as_display_1d(chunk)
        table.clear(columns=True)
        table.add_columns("")
        for i in range(int(a.shape[0])):
            table.add_row(_format_table_cell(a[i]), label=str(i))
        if n > PREVIEW_TABLE_1D_ROWS:
            table.add_row(_PREVIEW_ELLIPSIS, label=_PREVIEW_ELLIPSIS)
        return

    if dset.ndim == 2:
        r, c = int(dset.shape[0]), int(dset.shape[1])
        if min(r, c) == 1:
            if c == 1:
                vec = dset[:, 0]
            else:
                vec = dset[0, :]
            fill_browse_preview_datatable(vec, table)
            return
        br = min(PREVIEW_TABLE_2D_SIZE, r)
        bc = min(PREVIEW_TABLE_2D_SIZE, c)
        arr = np.asarray(dset[:br, :bc])
        nrows, ncols = int(arr.shape[0]), int(arr.shape[1])
        table.clear(columns=True)
        col_labels = [str(j) for j in range(ncols)]
        if c > bc:
            col_labels.append(_PREVIEW_ELLIPSIS)
        table.add_columns(*col_labels)
        for i in range(nrows):
            row = arr[i]
            cells = tuple(_format_table_cell(row[j]) for j in range(ncols))
            if c > bc:
                cells = cells + (_PREVIEW_ELLIPSIS,)
            table.add_row(*cells, label=str(i))
        if r > br:
            ell_cols = ncols + (1 if c > bc else 0)
            table.add_row(*((_PREVIEW_ELLIPSIS,) * ell_cols), label=_PREVIEW_ELLIPSIS)
        return

    raise ValueError(
        "browse preview table only supports plain 1D/2D, object 1D/2D, and compound 1D"
    )


def _dataset_path_element_count(h5file: h5py.File, path: str) -> int | None:
    """HDF5 element count without reading array data."""
    try:
        node = h5file[path]
        if isinstance(node, h5py.Dataset):
            return int(node.size)
    except Exception:
        pass
    return None


def should_use_1d_table(arr) -> bool:
    """Use a labeled-row DataTable (index + value column) for plain 1D data."""
    a = _as_display_1d(np.asarray(arr))
    return is_plain_1d_array(a) and a.size <= MAX_TABLE_ELEMENTS


def is_plain_2d_array(arr: np.ndarray) -> bool:
    """Non-record 2D arrays (numeric, fixed/object string grids)."""
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        return False
    return arr.dtype.fields is None


def should_use_2d_table(arr) -> bool:
    """2D plain array as a grid: column headers 0..n-1, row labels 0..m-1."""
    a = np.asarray(arr)
    if not is_plain_2d_array(a) or a.size > MAX_TABLE_ELEMENTS:
        return False
    if min(a.shape) == 0:
        return False
    # Vectors (N,1)/(1,N) use the 1D table after _as_display_1d in should_use_1d_table.
    if min(a.shape) == 1:
        return False
    ncols = int(a.shape[1])
    if ncols > MAX_TABLE_COLUMNS:
        return False
    return True


def should_use_ndarray_datatable(arr) -> bool:
    """Recarray uses ``update``; plain 1D/2D use ``update_1d`` / ``update_2d``."""
    return should_use_1d_table(arr) or should_use_2d_table(arr)


def _lazy_dataset_eligible(dset: h5py.Dataset) -> bool:
    """Large arrays: plain numeric, 1D vlen strings, or 1D compound — lazy table slices."""
    if int(dset.size) <= LAZY_THRESHOLD:
        return False
    if dset.ndim == 1 and dset.dtype == np.dtype("O"):
        return True
    if len(dset.dtype) == 0 and dset.dtype != np.dtype("O"):
        return True
    return _lazy_h5_structured_1d_datatable(dset)


def _lazy_h5_structured_1d_datatable(dset: h5py.Dataset) -> bool:
    """1D structured dtype with named fields (wide tables use a sliding field window)."""
    if dset.ndim != 1 or dset.dtype.fields is None:
        return False
    names = dset.dtype.names
    return bool(names)


def _lazy_h5_plain_1d_datatable(dset: h5py.Dataset) -> bool:
    """1D plain numeric, fixed strings, or HDF5 variable-length strings (numpy ``object``)."""
    if dset.ndim != 1 or dset.dtype.fields is not None:
        return False
    if dset.dtype == np.dtype("O"):
        return True
    return len(dset.dtype) == 0


def _lazy_h5_plain_2d_datatable(dset: h5py.Dataset) -> bool:
    """2D plain matrix; grid view with sliding row/column windows."""
    if len(dset.dtype) != 0 or dset.dtype == np.dtype("O") or dset.ndim != 2:
        return False
    r, c = int(dset.shape[0]), int(dset.shape[1])
    if min(r, c) == 0 or min(r, c) == 1:
        return False
    return c <= MAX_TABLE_COLUMNS


def _format_array_element(v) -> str:
    x = np.asarray(v)
    return np.array2string(
        x,
        max_line_width=10_000,
        suppress_small=False,
        separator=", ",
    )


def _format_table_cell(v: object) -> str:
    """Format one scalar for DataTable cells (1D, 2D, compound, text preview).

    Bytes/str pass through; structured/object/non-scalar arrays use
    ``array2string``. Plain scalars use space-padded ``format`` so ``+/-`` align
    in numeric columns.
    """
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).decode("utf-8", errors="replace")
    if isinstance(v, str):
        return v
    x = np.asarray(v)
    if x.dtype.names is not None or x.dtype == np.dtype("O"):
        return _format_array_element(v)
    if x.size != 1:
        return _format_array_element(v)
    e = x.item()
    if isinstance(e, bytes):
        return e.decode("utf-8", errors="replace")
    if isinstance(e, str):
        return e
    if np.issubdtype(x.dtype, np.bool_):
        return str(bool(e))
    if np.issubdtype(x.dtype, np.integer):
        return format(int(e), " d")
    if np.issubdtype(x.dtype, np.complexfloating):
        return _format_array_element(v)
    if np.issubdtype(x.dtype, np.inexact):
        return format(float(e), " f")
    return _format_array_element(v)


def format_1d_with_indices(arr: np.ndarray) -> str:
    """Format 1D data as ``index: value`` rows (browse preview); respects threshold/edgeitems."""
    a = np.asarray(arr)
    n = int(a.shape[0])
    threshold = _ND_THRESHOLD
    edgeitems = _ND_EDGEITEMS
    idxw = max(1, len(str(max(0, n - 1))))

    if n == 0:
        return ""

    def line(i: int) -> str:
        return f"{i:>{idxw}}: {_format_table_cell(a[i])}"

    if n <= threshold or n <= 2 * edgeitems:
        return "\n".join(line(i) for i in range(n))

    omitted = n - 2 * edgeitems
    parts = [line(i) for i in range(edgeitems)]
    parts.append(f"… ({omitted} elements omitted) …")
    parts.extend(line(i) for i in range(n - edgeitems, n))
    return "\n".join(parts)


def format_ndarray_text(
    arr: np.ndarray, *, line_width: int = _ND_LINEWIDTH_DEFAULT
) -> str:
    """Human-readable ndarray text; 1D uses indexed rows, other shapes use array2string."""
    a = _as_display_1d(np.asarray(arr))
    if is_plain_1d_array(a):
        return format_1d_with_indices(a)
    return np.array2string(
        a,
        max_line_width=line_width,
        threshold=_ND_THRESHOLD,
        edgeitems=_ND_EDGEITEMS,
        suppress_small=False,
    )


def format_1d_with_indices_lazy(dset: h5py.Dataset) -> str:
    """Indexed 1D text using only head/tail slices (no full vector load)."""
    n = int(dset.shape[0])
    edgeitems = _ND_EDGEITEMS
    idxw = max(1, len(str(max(0, n - 1))))
    if n == 0:
        return ""

    def line(i: int, val: object) -> str:
        return f"{i:>{idxw}}: {_format_table_cell(val)}"

    if n <= _ND_THRESHOLD or n <= 2 * edgeitems:
        a = np.asarray(dset[:n])
        return "\n".join(line(i, a[i]) for i in range(n))

    head = np.asarray(dset[:edgeitems])
    tail = np.asarray(dset[n - edgeitems : n])
    parts = [line(i, head[j]) for j, i in enumerate(range(edgeitems))]
    omitted = n - 2 * edgeitems
    parts.append(f"… ({omitted} elements omitted) …")
    parts.extend(line(i, tail[j]) for j, i in enumerate(range(n - edgeitems, n)))
    return "\n".join(parts)


def format_ndarray_text_lazy(dset: h5py.Dataset, *, line_width: int) -> str:
    """Abbreviated text for a large on-disk dataset without loading it entirely."""
    if dset.dtype.fields is not None:
        return "(unsupported dtype for text preview)"
    if dset.dtype == np.dtype("O"):
        if dset.ndim == 1:
            return format_1d_with_indices_lazy(dset)
        return "(unsupported dtype for text preview)"
    if dset.ndim == 1:
        return format_1d_with_indices_lazy(dset)
    if dset.ndim == 2:
        r, c = int(dset.shape[0]), int(dset.shape[1])
        rs, cs = min(8, r), min(12, c)
        corner = np.asarray(dset[:rs, :cs])
        body = np.array2string(
            corner,
            max_line_width=line_width,
            threshold=_ND_THRESHOLD,
            edgeitems=_ND_EDGEITEMS,
            suppress_small=False,
        )
        return f"shape {dset.shape} (top-left {rs}×{cs})\n{body}\n…"
    sl = tuple(slice(0, min(4, int(s))) for s in dset.shape)
    small = np.asarray(dset[sl])
    body = np.array2string(
        small,
        max_line_width=line_width,
        threshold=min(_ND_THRESHOLD, 100),
        edgeitems=min(_ND_EDGEITEMS, 2),
        suppress_small=False,
    )
    return f"shape {dset.shape}\n{body}\n…"


def is_plotable_lazy_dset(dset: h5py.Dataset) -> bool:
    if dset.dtype == np.dtype("O"):
        return False
    if dset.ndim == 2:
        return len(dset.dtype) == 0
    if dset.ndim != 1:
        return False
    if len(dset.dtype) == 0:
        return True
    return has_compound_plot_fields(dset.dtype)


def is_aggregatable_lazy_dset(dset: h5py.Dataset) -> bool:
    return (
        len(dset.dtype) == 0
        and dset.dtype != np.dtype("O")
        and np.issubdtype(dset.dtype, np.number)
        and int(dset.size) > 1
    )


def aggregate_h5_numeric_chunks(dset: h5py.Dataset) -> dict[str, float]:
    """Mean/var/std/min/max/L2 over chunks (float64 accumulators)."""
    n = int(dset.size)
    chunk = _AGG_CHUNK
    sum_x = 0.0
    sum_x2 = 0.0
    vmin = np.inf
    vmax = -np.inf

    linear: h5py.Dataset | None
    try:
        linear = dset.reshape((n,))
    except Exception:
        linear = None

    if linear is not None:
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            b = np.asarray(linear[start:end], dtype=np.float64)
            sum_x += float(b.sum())
            sum_x2 += float(np.dot(b, b))
            vmin = min(vmin, float(b.min()))
            vmax = max(vmax, float(b.max()))
    else:
        trailing = int(np.prod(dset.shape[1:])) if dset.ndim > 1 else 1
        rows_per_chunk = max(1, chunk // max(1, trailing))
        nrows = int(dset.shape[0])
        for r0 in range(0, nrows, rows_per_chunk):
            block = dset[r0 : min(r0 + rows_per_chunk, nrows)]
            b = np.asarray(block, dtype=np.float64).ravel()
            if b.size == 0:
                continue
            sum_x += float(b.sum())
            sum_x2 += float(np.dot(b, b))
            vmin = min(vmin, float(b.min()))
            vmax = max(vmax, float(b.max()))

    mean = sum_x / n
    var = max(0.0, sum_x2 / n - mean * mean)
    std = float(np.sqrt(var))
    l2 = float(np.sqrt(sum_x2))
    return {
        "mean": float(mean),
        "var": float(var),
        "std": std,
        "max": vmax,
        "min": vmin,
        "L2 norm": l2,
    }


def sample_dset_for_plot_1d(dset: h5py.Dataset) -> np.ndarray:
    n = int(dset.shape[0])
    if n <= _PLOT_MAX_1D:
        return np.asarray(dset[:])
    idx = np.linspace(0, n - 1, _PLOT_MAX_1D, dtype=np.intp)
    return np.asarray(dset[idx])


def sample_dset_for_plot_2d(dset: h5py.Dataset) -> np.ndarray:
    r, c = int(dset.shape[0]), int(dset.shape[1])
    mr, mc = min(_PLOT_MAX_2D, r), min(_PLOT_MAX_2D, c)
    if r <= _PLOT_MAX_2D and c <= _PLOT_MAX_2D:
        return np.asarray(dset[:, :])
    ri = np.linspace(0, r - 1, mr, dtype=np.intp)
    ci = np.linspace(0, c - 1, mc, dtype=np.intp)
    # h5py allows only 1D fancy indices; ``np.ix_`` is not supported on Dataset.
    rows = np.asarray(dset[ri])
    return rows[:, ci]


def add_escape_chars(string: str):
    return string.replace("[", r"\[")


def remove_escaped_chars(string: str):
    return string.replace(r"\[", "[")


def _text_item_highlight_ci(item: str, query: str) -> Text:
    """Case-insensitive match spans in ``item`` as styled :class:`~rich.text.Text` runs.

    Used for group find with ``OptionList(markup=False)``: markup strings would print
    literally; Rich ``Text`` + ``style=`` renders without the markup parser.
    """
    if not query:
        return Text(item)
    q_fold = query.casefold()
    if not q_fold or q_fold not in item.casefold():
        return Text(item)
    pat = re.escape(query)
    out = Text()
    pos = 0
    for m in re.finditer(f"(?i){pat}", item):
        if m.start() > pos:
            out.append(item[pos : m.start()])
        out.append(item[m.start() : m.end()], style="yellow")
        pos = m.end()
    if pos < len(item):
        out.append(item[pos:])
    return out


class MyOptionList(OptionList):
    BINDINGS = [
        Binding("down,j", "cursor_down", "Down", show=True),
        Binding("up,k", "cursor_up", "Up", show=True),
        Binding("G", "page_down", "Bottom", show=False),
        Binding("g", "page_up", "Top", show=False),
        Binding("ctrl+u,u", "half_page_up", "Half page up", show=False),
        Binding("ctrl+d,d", "half_page_down", "Half page down", show=False),
    ]

    def _move_half_page(self, direction: int) -> None:
        """Move the highlight by roughly half a viewport (same idea as ``OptionList._move_page``)."""
        if not self._options:
            return
        height = self.scrollable_content_region.height
        step = max(1, height // 2)
        y = clamp(
            self._index_to_line[self.highlighted or 0] + direction * step,
            0,
            len(self._lines) - 1,
        )
        option_index = self._lines[y][0]
        self.highlighted = _widget_navigation.find_next_enabled_no_wrap(
            candidates=self._options,
            anchor=option_index,
            direction=direction,
            with_anchor=True,
        )

    def action_half_page_up(self) -> None:
        self.refresh_bindings()
        if self.highlighted is None:
            self.action_first()
        else:
            self._move_half_page(-1)

    def action_half_page_down(self) -> None:
        self.refresh_bindings()
        if self.highlighted is None:
            self.action_last()
        else:
            self._move_half_page(1)

    def action_cursor_down(self) -> None:
        self.refresh_bindings()
        next_h = _widget_navigation.find_next_enabled_no_wrap(
            self.options,
            anchor=self.highlighted,
            direction=1,
            with_anchor=False,
        )
        if next_h is not None:
            self.highlighted = next_h

    def action_cursor_up(self) -> None:
        self.refresh_bindings()
        next_h = _widget_navigation.find_next_enabled_no_wrap(
            self.options,
            anchor=self.highlighted,
            direction=-1,
            with_anchor=False,
        )
        if next_h is not None:
            self.highlighted = next_h

    def check_action(self, action, parameters):
        if action in (
            "cursor_down",
            "cursor_up",
            "half_page_up",
            "half_page_down",
        ) and self.app.has_class("view-dataset"):
            return False
        else:
            return True


class ColumnContent(VerticalScroll):
    """Column which displays a dataset"""

    @property
    def allow_vertical_scroll(self) -> bool:
        """Disable outer scroll for browse DataTable preview — avoids clipping labels/rows."""
        if self.has_class("browse-preview-table"):
            return False
        return super().allow_vertical_scroll

    BINDINGS = [
        Binding("down,j", "scroll_down", "Down", show=True),
        Binding("up,k", "scroll_up", "Up", show=True),
        # Do not bind J/K here: they would scroll this VerticalScroll (the preview shell)
        # instead of the focused DataTable's ±10 row jump. Use j/k, or app J/K when the file list is focused.
        Binding("pageup,u", "page_up", "Page up", show=False),
        Binding("pagedown,d", "page_down", "Page down", show=False),
        Binding("G", "scroll_end", "Bottom", show=False),
        Binding("g", "scroll_home", "Top", show=False),
    ]

    def compose(self):
        self._content = Static(id="data", markup=False)
        self._plot = PlotextPlot(id="plot")
        self._df = LazyH5DataTable(id="dtable")
        yield self._content
        yield self._plot
        yield self._df

    def reprint(self):
        """Refresh dataset text or table content using ``H5TUIApp._data``."""
        app = self.app
        if getattr(app, "_lazy_mode", False):
            path = getattr(app, "_lazy_dset_path", None)
            if not path:
                return
            dset = app._file[path]
            lw = int(getattr(app, "_preview_line_width", _ND_LINEWIDTH_DEFAULT))
            if _lazy_h5_plain_1d_datatable(dset):
                extra = (
                    " — vlen UTF-8 strings, row window"
                    if dset.dtype == np.dtype("O")
                    else ""
                )
                _reprint_lazy_datatable(
                    self,
                    lambda: app.refresh_lazy_1d_datatable(cursor_row=0),
                    f"{_KEYS_HELP_TABLE_1D}{extra}",
                )
                return
            if _lazy_h5_structured_1d_datatable(dset):
                _reprint_lazy_datatable(
                    self,
                    lambda: app.refresh_lazy_1d_datatable(cursor_row=0),
                    "keys: j/k rows, h/l pan fields (wide compound), "
                    "H/L jump fields, J/K jump rows; y copy; esc quit",
                    timeout=6,
                )
                return
            if _lazy_h5_plain_2d_datatable(dset):
                _reprint_lazy_datatable(
                    self,
                    lambda: app.refresh_lazy_2d_datatable(cursor_row=0, cursor_col=0),
                    _KEYS_HELP_TABLE_2D,
                )
                return
            self.notify(
                "Abbreviated text preview (dataset too large for table).",
                severity="information",
                timeout=3,
            )
            self._content.update(format_ndarray_text_lazy(dset, line_width=lw))
            self.focus()
            return
        data = getattr(app, "_data", None)
        if data is None:
            return
        lw = int(getattr(app, "_preview_line_width", _ND_LINEWIDTH_DEFAULT))
        if is_dataframe(data):
            self.notify(_KEYS_HELP_TABLE_2D)
            self._df.update(data)
            self._df.focus()
        elif should_use_1d_table(data):
            self.notify(_KEYS_HELP_TABLE_1D)
            self._df.update_1d(_as_display_1d(np.asarray(data)))
            self._df.focus()
        elif should_use_2d_table(data):
            self.notify(_KEYS_HELP_TABLE_2D)
            self._df.update_2d(np.asarray(data))
            self._df.focus()
        else:
            self._content.update(format_ndarray_text(np.asarray(data), line_width=lw))

    def replot(self):
        """Plot data, currently only supports 1D and 2D data"""
        app = self.app
        plt = self._plot.plt
        if getattr(app, "_lazy_mode", False):
            path = getattr(app, "_lazy_dset_path", None)
            if not path:
                return
            dset = app._file[path]
            if not is_plotable_lazy_dset(dset):
                return
            if dset.dtype.fields is not None and dset.ndim == 1:
                data = sample_dset_for_plot_1d(dset)
                names = dtype_numeric_field_names(dset.dtype)
                if not names:
                    return
                yk = getattr(app, "_compound_plot_y", "") or names[0]
                if yk not in names:
                    yk = names[0]
                hist_field: str | None = None
                if bool(getattr(app, "_plot_histogram", False)):
                    hist_field = app._compound_histogram_value_field_for_preview()
                _plot_preview_compound_1d(
                    plt,
                    data,
                    histogram=bool(getattr(app, "_plot_histogram", False)),
                    compound_x=str(
                        getattr(app, "_compound_plot_x", COMPOUND_PLOT_INDEX)
                    ),
                    compound_y=str(yk),
                    log_x=bool(getattr(app, "_compound_plot_log_x", False)),
                    log_y=bool(getattr(app, "_compound_plot_log_y", False)),
                    histogram_value_field=hist_field,
                )
                return
            _plotext_reset_linear(plt)
            if dset.ndim == 1:
                data = sample_dset_for_plot_1d(dset)
                n = int(dset.shape[0])
                x = np.linspace(0, n - 1, data.shape[0])
                _plot_preview_plain_1d(
                    plt,
                    data,
                    histogram=bool(getattr(app, "_plot_histogram", False)),
                    x=x,
                )
            else:
                _plot_preview_2d(plt, sample_dset_for_plot_2d(dset))
            return
        raw = getattr(self.app, "_data", None)
        if raw is None:
            return
        data = np.squeeze(raw)
        app = self.app
        plt = self._plot.plt
        if data.dtype.fields is not None and data.ndim == 1:
            if not has_compound_plot_fields(data.dtype):
                return
            names = dtype_numeric_field_names(data.dtype)
            if not names:
                return
            yk = getattr(app, "_compound_plot_y", "") or names[0]
            if yk not in names:
                yk = names[0]
            hist_field: str | None = None
            if bool(getattr(app, "_plot_histogram", False)):
                hist_field = app._compound_histogram_value_field_for_preview()
            _plot_preview_compound_1d(
                plt,
                data,
                histogram=bool(getattr(app, "_plot_histogram", False)),
                compound_x=str(getattr(app, "_compound_plot_x", COMPOUND_PLOT_INDEX)),
                compound_y=str(yk),
                log_x=bool(getattr(app, "_compound_plot_log_x", False)),
                log_y=bool(getattr(app, "_compound_plot_log_y", False)),
                histogram_value_field=hist_field,
            )
            return
        if not is_plotable(data):
            return
        _plotext_reset_linear(plt)
        if data.ndim == 1:
            _plot_preview_plain_1d(
                plt,
                data,
                histogram=bool(getattr(app, "_plot_histogram", False)),
                x=np.arange(data.shape[0]),
            )
        elif data.ndim == 2:
            _plot_preview_2d(plt, data)


class H5TUIApp(App):
    """Simple tui application for displaying and navigating h5 files"""

    theme = Reactive("nord")

    BINDINGS = [
        Binding("i", "toggle_dark", "Toggle dark mode", show=False),
        Binding("q", "quit_contextual", "Close", show=False),
        Binding("Q", "quit_application", "Quit", show=False),
        Binding("escape", "exit_dataset_view", "Close", show=False),
        Binding("left,h", "goto_parent", "Back", show=True),
        Binding("right,l", "goto_child", "Select", show=True),
        Binding("a", "view_attrs", "Attributes", show=True),
        Binding("p", "toggle_plot", "Plot", show=True),
        Binding("P", "toggle_plot_histogram", "Histogram", show=True),
        Binding("s", "save_plot", "Save plot", show=True),
        Binding("A", "aggregate_data", "Summary", show=True),
        Binding(":", "jump_to_line", "Jump to line", show=False),
        Binding("/", "group_find", "Find in group", show=False),
        Binding("J", "scroll_preview_down", "Scroll preview", show=False),
        Binding("K", "scroll_preview_up", "Scroll preview", show=False),
    ]
    CSS_PATH = "h5tui.tcss"
    TITLE = "h5tui"

    def __init__(self, fname):
        super().__init__()

        self._fname = fname
        try:
            self._file = h5py.File(fname, "r")
        except OSError as e:
            print(f"Could not open HDF5 file: {e}", file=sys.stderr)
            sys.exit(1)

        self.title = f"h5tui · {os.path.basename(fname)}"

        self._cur_dir = str(self._file.name)

        # One index per descent into a group (keyboard ``l``): used to restore the
        # highlighted row in each ancestor when backing out with ``h``. Cleared when
        # the path jumps via the Parent column (arbitrary branch).
        self._browse_highlight_stack: list[int] = []
        self._preview_line_width = _ND_LINEWIDTH_DEFAULT
        self._data = None

        self._refresh_guard = False
        # When True, 1D plot view shows a value histogram instead of index vs value.
        self._plot_histogram = False
        # 1D compound: line/histogram axes (``COMPOUND_PLOT_INDEX`` = row index for X).
        self._compound_plot_x = COMPOUND_PLOT_INDEX
        self._compound_plot_y = ""
        self._compound_plot_log_x = False
        self._compound_plot_log_y = False
        # Browse-mode compound plot: sample held until ``SavePlotScreen`` (plot_settings_only) confirms.
        self._pending_browse_compound_plot: tuple[np.ndarray, str] | None = None
        # Browse-mode compound histogram: sample + chosen field after ``histogram_settings_only`` confirms.
        self._pending_browse_histogram_sample: np.ndarray | None = None
        self._compound_browse_histogram_field: str | None = None
        # HDF5 path for the dataset opened in full view (basename used for save-as default).
        self._open_dataset_path: str | None = None
        # Large-dataset double-confirm: same key as ``_large_op_arm`` + second press runs op.
        self._large_op_arm: tuple | None = None
        self._large_confirm_timer = None
        # Lazy HDF5 view: keep path + window state; avoid ``dset[...]`` for huge arrays.
        self._lazy_mode = False
        self._lazy_dset_path: str | None = None
        self._lazy_1d_windowed = False
        self._lazy_1d_total = 0
        self._lazy_1d_window_start = 0
        self._lazy_2d_windowed = False
        self._lazy_2d_total_rows = 0
        self._lazy_2d_total_cols = 0
        self._lazy_2d_window_row = 0
        self._lazy_2d_window_col = 0
        self._lazy_structured_1d = False
        self._lazy_struct_field_start = 0
        self._browse_find_open = False
        self._group_find_query = ""
        self._browse_find_saved_highlight = 0

    def on_unmount(self) -> None:
        self._cancel_large_confirm_timer()
        self._file.close()

    def _header_content(
        self,
        h5_path: str,
        *,
        dataset_meta: str | None = None,
    ) -> Content:
        if dataset_meta is None:
            return Content.from_markup("Path: [$accent]$path[/]", path=h5_path)
        return Content.from_markup(
            "Path: [$accent]$path[/]\nDataset: [$accent]$meta[/]",
            path=h5_path,
            meta=escape(dataset_meta),
        )

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

        self._header_widget = Static(self._header_content("/"), id="header")
        yield self._header_widget
        with Horizontal(id="browser"):
            self._parent_list = MyOptionList(id="dirs-parent", markup=False)
            self._current_list = MyOptionList(id="dirs-current", markup=False)
            self._preview = ColumnContent(id="content")
            yield self._parent_list
            yield self._current_list
            yield self._preview

    def on_mount(self) -> None:
        self._preview_line_width = max(40, self.size.width)
        self._parent_list.border_title = "Parent"
        self._current_list.border_title = "Current"
        self._preview.border_title = "Preview"
        self.refresh_parent_column()
        self.refresh_current_column(0)
        self.refresh_preview()
        self._current_list.focus()

    def on_resize(self, event: Resize) -> None:
        self._preview_line_width = max(40, event.size.width)

    def _row_icon_for_group_child(
        self, group: h5py.Group, dir_path: str, name: str
    ) -> str:
        """Leading icon: link (soft/external), else group vs dataset; broken soft uses link icon."""
        try:
            lnk = group.get(name, getlink=True)
        except Exception:
            return "🔗  " if UNICODE_SUPPORT else "(link) "
        if isinstance(lnk, (h5py.SoftLink, h5py.ExternalLink)):
            return "🔗  " if UNICODE_SUPPORT else "(link) "
        try:
            h5elem = group[name]
        except Exception:
            return "🔗  " if UNICODE_SUPPORT else "(link) "
        if UNICODE_SUPPORT:
            if isinstance(h5elem, h5py.Group):
                return "📁  "
            if isinstance(h5elem, h5py.Dataset):
                return "📊  "
        else:
            if isinstance(h5elem, h5py.Group):
                return "(Group)    "
            if isinstance(h5elem, h5py.Dataset):
                return "(DataSet)  "
        return "📄  " if UNICODE_SUPPORT else "(? )    "

    def _metadata_line_for_item(
        self, dir_path: str, item: str, canonical_map: dict[str, str]
    ) -> str:
        group = self._file[dir_path]
        if not isinstance(group, h5py.Group):
            return item
        prefix = self._row_icon_for_group_child(group, dir_path, item)
        arrow = _link_arrow_suffix(group, dir_path, item, canonical_map)
        return f"{prefix}{item}{arrow}    {self.build_attr_str(dir_path, item)}"

    def _metadata_line_for_group_find(
        self,
        dir_path: str,
        item: str,
        canonical_map: dict[str, str],
        query: str,
    ) -> Text:
        group = self._file[dir_path]
        if not isinstance(group, h5py.Group):
            return Text(item)
        prefix = self._row_icon_for_group_child(group, dir_path, item)
        arrow = _link_arrow_suffix(group, dir_path, item, canonical_map)
        rest = f"{arrow}    {self.build_attr_str(dir_path, item)}"
        middle = _text_item_highlight_ci(item, query) if query else Text(item)
        line = Text()
        line.append(prefix)
        line.append_text(middle)
        line.append(rest)
        return line

    def get_selected_itemname(self):
        """Return the HDF5 child name for the highlighted row in Current, or None."""
        highlighted = self._current_list.highlighted
        if highlighted is None:
            return None
        items = self.get_dir_content(self._cur_dir)
        if highlighted < 0 or highlighted >= len(items):
            return None
        return items[highlighted]

    def get_selected_dataset_path(self) -> str | None:
        """POSIX path to the highlighted item if it is a :class:`h5py.Dataset`, else ``None``."""
        name = self.get_selected_itemname()
        if name is None:
            return None
        path = h5_child_path(self._cur_dir, name)
        try:
            if path in self._file and isinstance(self._file[path], h5py.Dataset):
                return path
        except Exception:
            pass
        return None

    def has_attr(self):
        """Return if the currently selected item has attributes"""
        selected_item = self.get_selected_itemname()
        if selected_item is None:
            return False
        return self.build_attr_str(self._cur_dir, selected_item) != ""

    def build_attr_str(self, dir_path: str, elem: str):
        """Creates the has attributes string (▼ + num attrs)"""
        path = h5_child_path(dir_path, elem)
        try:
            h5elem = self._file[path]
        except Exception:
            return ""
        num_attrs = len(h5elem.attrs)
        if num_attrs > 0:
            return f"▼ ({num_attrs})"
        else:
            return ""

    def add_dir_metadata_for(self, dir_path: str):
        grp = self._file[dir_path]
        if not isinstance(grp, h5py.Group):
            return []
        items = list(grp.keys())
        canonical_map = _build_link_canonical_name_map(grp, dir_path)
        return [
            self._metadata_line_for_item(dir_path, item, canonical_map)
            for item in items
        ]

    def add_dir_metadata(self, dir_path=None):
        dir_path = dir_path or self._cur_dir
        return self.add_dir_metadata_for(dir_path)

    def get_dir_content(self, dir):
        """Return contents of current path"""
        return list(self._file[dir].keys())

    def refresh_parent_column(self) -> None:
        pdir = h5_parent_path(self._cur_dir)
        if pdir is None:
            self._parent_list.border_title = "Parent · /"
            self._parent_list.clear_options()
            return
        self._parent_list.border_title = f"Parent · {pdir}"
        items = self.get_dir_content(pdir)
        opts = self.add_dir_metadata_for(pdir)
        self._parent_list.clear_options()
        self._parent_list.add_options(opts)
        target = os.path.basename(self._cur_dir.rstrip("/"))
        try:
            hi = items.index(target)
        except ValueError:
            hi = 0 if items else None
        self._refresh_guard = True
        try:
            if hi is not None and items:
                self._parent_list.highlighted = hi
        finally:
            self._refresh_guard = False

    def refresh_current_column(self, prev_highlighted: int = 0) -> None:
        self._current_list.border_title = "Current"
        opts = self.add_dir_metadata()
        self._current_list.clear_options()
        self._current_list.add_options(opts)
        n = len(opts)
        if n == 0:
            self._current_list.highlighted = None
        else:
            self._current_list.highlighted = min(prev_highlighted, n - 1)

    def _refresh_current_group_find_list(self, query: str) -> None:
        """Rebuild Current column with Rich Text (styled runs); jump to first match."""
        grp = self._file[self._cur_dir]
        if not isinstance(grp, h5py.Group):
            self._current_list.clear_options()
            self._current_list.highlighted = None
            return
        items = list(grp.keys())
        canonical_map = _build_link_canonical_name_map(grp, self._cur_dir)
        opts = [
            self._metadata_line_for_group_find(
                self._cur_dir, item, canonical_map, query
            )
            for item in items
        ]
        prev_highlight = self._current_list.highlighted
        self._refresh_guard = True
        try:
            self._current_list.clear_options()
            self._current_list.add_options(opts)
        finally:
            self._refresh_guard = False
        n = len(opts)
        q_fold = query.casefold() if query else ""
        if query and q_fold:
            first = next(
                (i for i, name in enumerate(items) if q_fold in name.casefold()),
                None,
            )
            if first is not None:
                self._current_list.highlighted = first
            elif n:
                # No matches: keep the row that was selected before this refresh.
                if prev_highlight is not None and 0 <= prev_highlight < n:
                    self._current_list.highlighted = prev_highlight
                else:
                    saved = max(0, self._browse_find_saved_highlight)
                    self._current_list.highlighted = min(saved, n - 1)
            else:
                self._current_list.highlighted = None
        elif n:
            saved = max(0, self._browse_find_saved_highlight)
            self._current_list.highlighted = min(saved, n - 1)
        else:
            self._current_list.highlighted = None
        if not self.has_class("view-dataset"):
            self.refresh_preview()

    def apply_group_find_query(self, query: str) -> None:
        if not self._browse_find_open:
            return
        self._group_find_query = query
        self._refresh_current_group_find_list(query)

    def finish_group_find(self, *, cancelled: bool) -> None:
        self._browse_find_open = False
        self._group_find_query = ""
        if cancelled:
            prev = max(0, self._browse_find_saved_highlight)
            self.refresh_current_column(prev)
        else:
            hi = self._current_list.highlighted
            if hi is None:
                hi = max(0, self._browse_find_saved_highlight)
            self.refresh_current_column(hi)
        self.update_header(self._header_content(self._cur_dir))
        if not self.has_class("view-dataset"):
            self._current_list.focus()

    def action_group_find(self) -> None:
        if isinstance(self.screen, _MODAL_SCREENS_ALL):
            return
        if self.has_class("view-dataset"):
            self.notify(
                "Close the dataset view (esc) to search the group list",
                severity="information",
                timeout=2,
            )
            return
        self._browse_find_saved_highlight = (
            self._current_list.highlighted
            if self._current_list.highlighted is not None
            else 0
        )
        self._browse_find_open = True
        self._group_find_query = ""
        self.push_screen(GroupFindScreen())

    def _dataset_preview_metadata_block(self, dset: h5py.Dataset) -> str:
        """Browse metadata: plain arrays show shape/dtype; structured show same summary line as header."""
        cd = _dataset_compression_detail(dset)
        if dset.dtype.fields is not None:
            lines = [
                _structured_dataset_summary_line(dset),
                f"size: {_dataset_size_display(dset)}",
            ]
        else:
            lines = [
                f"shape: {_plain_dataset_shape_display(dset)}",
                f"dtype: {dset.dtype}",
                f"size: {_dataset_size_display(dset)}",
            ]
        if dset.chunks is not None:
            lines.append(f"chunks: {dset.chunks}")
        if cd:
            lines.append(f"compression: {cd}")
        lines.append("")
        return "\n".join(lines)

    def _dataset_preview_block(self, dset: h5py.Dataset) -> str:
        lw = int(self._preview_line_width)
        lines = [self._dataset_preview_metadata_block(dset)]
        if dset.size == 0:
            lines.append("(empty)")
            return "\n".join(lines)
        if len(dset.dtype) != 0:
            lines.append("(no browse preview — press l for full view)")
            return "\n".join(lines)
        try:
            if dset.size <= PREVIEW_BROWSE_TEXT_MAX_ELEMENTS:
                arr = np.asarray(dset[...])
                lines.append(format_ndarray_text(arr, line_width=lw))
                return "\n".join(lines)
            if dset.ndim == 1:
                n = min(24, dset.shape[0])
                lines.append(format_ndarray_text(np.asarray(dset[:n]), line_width=lw))
                lines.append("… (press l for full view)")
            elif dset.ndim == 2:
                r = min(6, dset.shape[0])
                c = min(10, dset.shape[1])
                lines.append(
                    np.array2string(
                        np.asarray(dset[:r, :c]),
                        max_line_width=lw,
                        threshold=200,
                        edgeitems=2,
                        suppress_small=False,
                    )
                )
                lines.append("… (press l for full view)")
            else:
                lines.append(
                    "(3D+ array — no browse slice preview; press l for full view)"
                )
        except Exception as e:
            lines.append(f"(could not read data: {e})")
        return "\n".join(lines)

    def _clear_detail_view_state(self) -> None:
        """Leave full dataset / plot / table view and restore default printing."""
        self._data = None
        self._open_dataset_path = None
        self._pending_browse_compound_plot = None
        self._pending_browse_histogram_sample = None
        self._compound_browse_histogram_field = None
        self._plot_histogram = False
        self._lazy_mode = False
        self._lazy_dset_path = None
        self._lazy_1d_windowed = False
        self._lazy_1d_total = 0
        self._lazy_1d_window_start = 0
        self._lazy_2d_windowed = False
        self._lazy_2d_total_rows = 0
        self._lazy_2d_total_cols = 0
        self._lazy_2d_window_row = 0
        self._lazy_2d_window_col = 0
        self._lazy_structured_1d = False
        self._lazy_struct_field_start = 0
        self.remove_class("view-dataset")
        self.remove_class("view-plot")
        self.remove_class("view-dtable")
        self.remove_class("preview-maximized")
        self._preview.remove_class("browse-preview-table")

    def refresh_preview(self) -> None:
        """Update the preview pane from the Current column selection (browse mode)."""
        if self.has_class("view-dataset"):
            return
        if self.has_class("view-plot") and not self.has_class("view-dataset"):
            self.remove_class("view-plot")
            self._plot_histogram = False
            self._data = None
            self._pending_browse_compound_plot = None
            self._pending_browse_histogram_sample = None
            self._compound_browse_histogram_field = None
        self._preview.remove_class("empty-group-preview")
        self._preview.remove_class("browse-preview-table")
        name = self.get_selected_itemname()
        if name is None:
            self._preview._content.update("(empty directory)")
            return
        path = h5_child_path(self._cur_dir, name)
        try:
            node = self._file[path]
        except Exception as e:
            self._preview._content.update(f"(could not open item: {e})")
            return
        if isinstance(node, h5py.Group):
            try:
                keys = list(node.keys())
            except Exception as e:
                self._preview._content.update(f"(could not read group: {e})")
                return
            if not keys:
                self._preview.add_class("empty-group-preview")
                self._preview._content.update("\nNo Content")
            else:
                shown = keys[:80]
                more = len(keys) - len(shown)
                cm = _build_link_canonical_name_map(node, path)
                lines = [self._metadata_line_for_item(path, k, cm) for k in shown]
                text = "\n".join(lines)
                if more > 0:
                    text += f"\n… ({more} more)"
                self._preview._content.update(text)
        elif isinstance(node, h5py.Dataset):
            if _browse_preview_datatable_eligible(node):
                self._preview.add_class("browse-preview-table")
                self._preview._content.update(
                    self._dataset_preview_metadata_block(node)
                )
                try:
                    fill_browse_preview_datatable(node, self._preview._df)
                    self._preview.scroll_x = 0
                    self._preview.scroll_y = 0
                    self._preview._df.scroll_x = 0
                    self._preview._df.scroll_y = 0
                except Exception as e:
                    self._preview.remove_class("browse-preview-table")
                    self._preview._content.update(
                        self._dataset_preview_metadata_block(node)
                        + f"(could not read data: {e})"
                    )
            else:
                self._preview._content.update(self._dataset_preview_block(node))

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if self._refresh_guard:
            return
        oid = event.option_list.id
        if oid == "dirs-parent":
            pdir = h5_parent_path(self._cur_dir)
            if pdir is None:
                return
            keys = self.get_dir_content(pdir)
            idx = event.option_index
            if idx < 0 or idx >= len(keys):
                return
            new_dir = h5_child_path(pdir, keys[idx])
            if new_dir == self._cur_dir:
                return
            self._cur_dir = new_dir
            self._browse_highlight_stack.clear()
            self._clear_detail_view_state()
            self.refresh_current_column(0)
            self.refresh_parent_column()
            self.refresh_preview()
            self.update_header(self._header_content(self._cur_dir))
            self.refresh_bindings()
            return
        if oid == "dirs-current":
            # Mouse (or other) moves the highlight while a dataset is open: leave
            # full view so the preview matches the new selection.
            if self.has_class("view-dataset"):
                self._clear_detail_view_state()
            self.refresh_preview()

    def _reset_compound_plot_defaults_for_dtype(self, dtype: np.dtype) -> None:
        names = dtype_numeric_field_names(dtype)
        if not names:
            return
        self._compound_plot_x = COMPOUND_PLOT_INDEX
        self._compound_plot_y = names[0]
        self._compound_plot_log_x = False
        self._compound_plot_log_y = False

    def _is_compound_dataset_view_for_plot(self) -> bool:
        """Full dataset view (``l``): structured 1D with numeric fields for Plotext."""
        if not self.has_class("view-dataset"):
            return False
        if getattr(self, "_lazy_mode", False) and self._lazy_dset_path:
            d = self._file[self._lazy_dset_path]
            return (
                d.dtype.fields is not None
                and d.ndim == 1
                and has_compound_plot_fields(d.dtype)
            )
        if self._data is None:
            return False
        a = np.asarray(self._data)
        return (
            a.dtype.fields is not None
            and a.ndim == 1
            and has_compound_plot_fields(a.dtype)
        )

    def _sample_and_name_compound_plot_dialog(
        self,
    ) -> tuple[np.ndarray, str] | None:
        """Array sample and short name for compound plot settings modal."""
        path = self._open_dataset_path
        if path is None:
            return None
        base = os.path.basename(path.rstrip("/")) or "plot"
        if getattr(self, "_lazy_mode", False) and self._lazy_dset_path:
            dset = self._file[self._lazy_dset_path]
            if (
                dset.dtype.fields is None
                or dset.ndim != 1
                or not has_compound_plot_fields(dset.dtype)
            ):
                return None
            return sample_dset_for_plot_1d(dset), base
        if self._data is None:
            return None
        a = np.asarray(self._data)
        if (
            a.dtype.fields is None
            or a.ndim != 1
            or not has_compound_plot_fields(a.dtype)
        ):
            return None
        return a, base

    def _on_compound_plot_settings_confirmed(self) -> None:
        """After ``Show plot`` on compound settings modal: show Plotext pane."""
        pend = self._pending_browse_compound_plot
        self._pending_browse_compound_plot = None
        if pend is not None:
            self._data = pend[0]
        self._plot_histogram = False
        self._compound_browse_histogram_field = None
        self._preview.remove_class("browse-preview-table")
        self.add_class("view-plot")
        self._preview.replot()
        if pend is not None:
            self._current_list.focus()
        else:
            self._preview.focus()
        self.refresh_bindings()

    def _on_compound_plot_settings_cancelled(self) -> None:
        self._pending_browse_compound_plot = None

    def _on_compound_histogram_browse_confirmed(self) -> None:
        """After ``Show histogram`` on browse compound modal: show histogram in preview."""
        sample = self._pending_browse_histogram_sample
        self._pending_browse_histogram_sample = None
        if sample is not None:
            self._data = sample
        self._plot_histogram = True
        opening = not self.has_class("view-plot")
        if opening:
            self.notify("Histogram…", timeout=1.5)
        self._preview.remove_class("browse-preview-table")
        self.add_class("view-plot")
        self._preview.replot()
        self._current_list.focus()
        self.refresh_bindings()

    def _on_compound_histogram_browse_cancelled(self) -> None:
        self._pending_browse_histogram_sample = None
        self._compound_browse_histogram_field = None

    def _compound_histogram_value_field_for_preview(self) -> str | None:
        """Histogram column: DataTable cursor (full view) or browse modal field."""
        c = self._compound_histogram_field_from_cursor()
        if c is not None:
            return c
        if (
            self.has_class("view-plot")
            and not self.has_class("view-dataset")
            and bool(getattr(self, "_plot_histogram", False))
        ):
            bf = getattr(self, "_compound_browse_histogram_field", None)
            return str(bf) if bf else None
        return None

    def update_content(self, path):
        self._preview.remove_class("browse-preview-table")
        self._preview.remove_class("empty-group-preview")
        self.remove_class("view-plot")
        self._plot_histogram = False
        try:
            node = self._file[path]
            if not isinstance(node, h5py.Dataset):
                self.notify(f"Not a dataset: {path}", severity="error", timeout=8)
                return
            dset = node
        except Exception as e:
            self.notify(f"Could not read dataset: {e}", severity="error", timeout=8)
            return

        lazy = _lazy_dataset_eligible(dset)
        if lazy:
            self._lazy_mode = True
            self._lazy_dset_path = path
            self._data = None
        else:
            try:
                dset_data = dset[...]
            except Exception as e:
                self.notify(f"Could not read dataset: {e}", severity="error", timeout=8)
                return
            self._lazy_mode = False
            self._lazy_dset_path = None
            self._lazy_structured_1d = False
            self._data = dset_data
            self._lazy_1d_windowed = False
            self._lazy_2d_windowed = False

        self._open_dataset_path = path

        self.add_class("view-dataset")
        self.add_class("preview-maximized")
        if lazy:
            self._lazy_structured_1d = False
            if _lazy_h5_plain_1d_datatable(dset):
                self.add_class("view-dtable")
                self._lazy_1d_windowed = True
                self._lazy_2d_windowed = False
                self._lazy_1d_total = int(dset.shape[0])
                self._lazy_1d_window_start = 0
            elif _lazy_h5_structured_1d_datatable(dset):
                self._lazy_structured_1d = True
                self._lazy_struct_field_start = 0
                self.add_class("view-dtable")
                self._lazy_1d_windowed = True
                self._lazy_2d_windowed = False
                self._lazy_1d_total = int(dset.shape[0])
                self._lazy_1d_window_start = 0
            elif _lazy_h5_plain_2d_datatable(dset):
                self.add_class("view-dtable")
                self._lazy_2d_windowed = True
                self._lazy_1d_windowed = False
                self._lazy_2d_total_rows = int(dset.shape[0])
                self._lazy_2d_total_cols = int(dset.shape[1])
                self._lazy_2d_window_row = 0
                self._lazy_2d_window_col = 0
            else:
                self._lazy_1d_windowed = False
                self._lazy_2d_windowed = False
        elif is_dataframe(self._data):
            self.add_class("view-dtable")
        elif should_use_ndarray_datatable(self._data):
            self.add_class("view-dtable")

        if (
            dset.dtype.fields is not None
            and dset.ndim == 1
            and has_compound_plot_fields(dset.dtype)
        ):
            self._reset_compound_plot_defaults_for_dtype(dset.dtype)

        self._preview.reprint()

        self.update_header(
            self._header_content(
                path,
                dataset_meta=format_dataset_header_metadata(dset),
            )
        )
        # Focus DataTable when it is the active view; focusing the parent ColumnContent
        # would steal focus from the table (requires mouse click to use j/k on rows).
        if self.has_class("view-dtable"):
            self._preview._df.focus()
        else:
            self._preview.focus()

    def refresh_lazy_1d_datatable(
        self,
        *,
        cursor_row: int | None = None,
        cursor_col: int | None = None,
    ) -> None:
        """Load ``LAZY_DATATABLE_WINDOW_SIZE`` rows from ``_lazy_1d_window_start``."""
        if not self._lazy_dset_path:
            return
        dset = self._file[self._lazy_dset_path]
        n = self._lazy_1d_total
        w = LAZY_DATATABLE_WINDOW_SIZE
        base = max(0, min(self._lazy_1d_window_start, max(0, n - w)))
        self._lazy_1d_window_start = base
        end = min(base + w, n)
        chunk = np.asarray(dset[base:end])
        if getattr(self, "_lazy_structured_1d", False):
            names = tuple(dset.dtype.names or ())
            nf = len(names)
            fv = min(LAZY_STRUCT_FIELD_WINDOW, nf)
            mx_fs = max(0, nf - fv)
            fs = max(
                0,
                min(int(getattr(self, "_lazy_struct_field_start", 0)), mx_fs),
            )
            self._lazy_struct_field_start = fs
            field_names = names[fs : fs + fv]
            self._preview._df.update_structured_1d_window(
                chunk,
                base,
                field_names,
                field_start=fs,
                n_fields_total=nf,
            )
            if cursor_row is not None and self._preview._df.row_count > 0:
                cr = max(0, min(int(cursor_row), self._preview._df.row_count - 1))
                ncols = len(self._preview._df.columns)
                if cursor_col is not None and ncols > 0:
                    ccm = max(0, min(int(cursor_col), ncols - 1))
                    self._preview._df.move_cursor(row=cr, column=ccm, scroll=True)
                else:
                    self._preview._df.move_cursor(row=cr, column=0, scroll=True)
            return
        self._preview._df.border_title = ""
        self._preview._df.update_1d_window(chunk, base)
        if cursor_row is not None and self._preview._df.row_count > 0:
            cr = max(0, min(int(cursor_row), self._preview._df.row_count - 1))
            self._preview._df.move_cursor(row=cr, column=0, scroll=True)

    def lazy_1d_set_global_index(self, global_idx: int) -> None:
        """Scroll the window so global row index ``global_idx`` is visible and selected."""
        n = self._lazy_1d_total
        w = LAZY_DATATABLE_WINDOW_SIZE
        g = max(0, min(int(global_idx), n - 1))
        if n <= w:
            self._lazy_1d_window_start = 0
            cur = g
        else:
            self._lazy_1d_window_start = min(max(0, g - w + 1), n - w)
            cur = g - self._lazy_1d_window_start
        cc = None
        if getattr(self, "_lazy_structured_1d", False):
            cc = self._preview._df.cursor_coordinate[1]
        self.refresh_lazy_1d_datatable(cursor_row=cur, cursor_col=cc)

    def refresh_lazy_2d_datatable(
        self, *, cursor_row: int | None = None, cursor_col: int | None = None
    ) -> None:
        """Load a ``LAZY_2D_WINDOW_ROWS`` × ``LAZY_2D_WINDOW_COLS`` slice from disk."""
        if not self._lazy_dset_path:
            return
        dset = self._file[self._lazy_dset_path]
        nr = self._lazy_2d_total_rows
        nc = self._lazy_2d_total_cols
        wr, wc = LAZY_2D_WINDOW_ROWS, LAZY_2D_WINDOW_COLS
        br = max(0, min(self._lazy_2d_window_row, max(0, nr - 1)))
        bc = max(0, min(self._lazy_2d_window_col, max(0, nc - 1)))
        if nr > wr:
            br = min(br, nr - wr)
        else:
            br = 0
        if nc > wc:
            bc = min(bc, nc - wc)
        else:
            bc = 0
        self._lazy_2d_window_row = br
        self._lazy_2d_window_col = bc
        r2 = min(wr, nr - br)
        c2 = min(wc, nc - bc)
        chunk = np.asarray(dset[br : br + r2, bc : bc + c2])
        self._preview._df.update_2d_window(chunk, br, bc)
        if cursor_row is not None and self._preview._df.row_count > 0:
            cr = max(0, min(int(cursor_row), self._preview._df.row_count - 1))
            ncols = len(self._preview._df.columns)
            cc = max(
                0, min(int(cursor_col if cursor_col is not None else 0), ncols - 1)
            )
            self._preview._df.move_cursor(row=cr, column=cc, scroll=True)

    def lazy_2d_set_global_cell(self, global_row: int, global_col: int) -> None:
        """Scroll both axes so global cell ``(global_row, global_col)`` is selected."""
        nr = self._lazy_2d_total_rows
        nc = self._lazy_2d_total_cols
        wr, wc = LAZY_2D_WINDOW_ROWS, LAZY_2D_WINDOW_COLS
        gr = max(0, min(int(global_row), nr - 1))
        gc = max(0, min(int(global_col), nc - 1))
        if nr <= wr:
            br, cr = 0, gr
        else:
            br = min(max(0, gr - wr + 1), nr - wr)
            cr = gr - br
        if nc <= wc:
            bc, cc = 0, gc
        else:
            bc = min(max(0, gc - wc + 1), nc - wc)
            cc = gc - bc
        self._lazy_2d_window_row = br
        self._lazy_2d_window_col = bc
        self.refresh_lazy_2d_datatable(cursor_row=cr, cursor_col=cc)

    def _jump_to_line_eligible(self) -> bool:
        """Plain 1D/2D DataTable (lazy windowed or small in-memory), not plot/recarray."""
        if not self.has_class("view-dataset") or not self.has_class("view-dtable"):
            return False
        if self.has_class("view-plot"):
            return False
        if getattr(self, "_lazy_mode", False) and self._lazy_dset_path:
            dset = self._file[self._lazy_dset_path]
            return (
                _lazy_h5_plain_1d_datatable(dset)
                or _lazy_h5_structured_1d_datatable(dset)
                or _lazy_h5_plain_2d_datatable(dset)
            )
        data = getattr(self, "_data", None)
        if data is None:
            return False
        return should_use_1d_table(data) or should_use_2d_table(data)

    def apply_jump_to_line_row(self, row_index: int) -> None:
        """Select global row ``row_index`` (0-based); 2D keeps the current column."""
        if not self._jump_to_line_eligible():
            return
        df = self._preview._df
        if getattr(self, "_lazy_mode", False) and self._lazy_dset_path:
            dset = self._file[self._lazy_dset_path]
            if _lazy_h5_plain_1d_datatable(dset) or _lazy_h5_structured_1d_datatable(
                dset
            ):
                n = self._lazy_1d_total
                if n <= 0:
                    return
                g = max(0, min(int(row_index), n - 1))
                self.lazy_1d_set_global_index(g)
                return
            if _lazy_h5_plain_2d_datatable(dset):
                nr = self._lazy_2d_total_rows
                if nr <= 0:
                    return
                gr = max(0, min(int(row_index), nr - 1))
                _, c = df.cursor_coordinate
                bc = self._lazy_2d_window_col
                gc = bc + c
                self.lazy_2d_set_global_cell(gr, gc)
                return
            return
        data = self._data
        if data is None:
            return
        if should_use_1d_table(data):
            a = _as_display_1d(np.asarray(data))
            n = int(a.shape[0])
            if n <= 0:
                return
            r = max(0, min(int(row_index), n - 1))
            df.move_cursor(row=r, column=0, scroll=True)
        elif should_use_2d_table(np.asarray(data)):
            arr = np.asarray(data)
            nrows = int(arr.shape[0])
            if nrows <= 0:
                return
            r = max(0, min(int(row_index), nrows - 1))
            _, cc = df.cursor_coordinate
            ncols = len(df.columns)
            cc = max(0, min(cc, ncols - 1))
            df.move_cursor(row=r, column=cc, scroll=True)

    def action_jump_to_line(self) -> None:
        """Open row jump dialog for plain 1D/2D dataset tables (``:``)."""
        if isinstance(self.screen, _MODAL_SCREENS_ALL):
            return
        if not self.has_class("view-dataset"):
            return
        if not self._jump_to_line_eligible():
            self.notify(
                "Jump to line is only available in 1D/2D DataTable view",
                severity="warning",
                timeout=2,
            )
            return
        self.push_screen(JumpToLineScreen())

    def update_header(self, header: Content) -> None:
        self._header_widget.update(header)

    def action_scroll_preview_down(self) -> None:
        """Nudge preview when file lists are focused (``J`` does not steal table jump)."""
        if (
            self.has_class("view-dtable")
            and self.has_class("view-dataset")
            and not self.has_class("view-plot")
        ):
            self._preview._df.action_jump_down()
        else:
            self._preview.scroll_down()

    def action_scroll_preview_up(self) -> None:
        if (
            self.has_class("view-dtable")
            and self.has_class("view-dataset")
            and not self.has_class("view-plot")
        ):
            self._preview._df.action_jump_up()
        else:
            self._preview.scroll_up()

    def check_action(self, action, parameters):
        if action in ("scroll_preview_down", "scroll_preview_up") and isinstance(
            self.screen, _MODAL_SCREENS_ALL
        ):
            # Attribute/save/summary modals use J/K for inner scroll.
            return False
        elif action in ("scroll_preview_down", "scroll_preview_up") and (
            self.focused is self._preview._df
        ):
            # DataTable owns J/K for ±10 row jump; do not run app handler.
            return False
        elif action in (
            "cursor_down",
            "cursor_up",
            "half_page_up",
            "half_page_down",
        ) and self.has_class("view-dataset"):
            return False
        elif (
            action in ("goto_parent", "goto_child")
            and self.focused is self._preview._df
        ):
            # Let MyDataTable handle h / l for column moves instead of app navigation.
            return False
        elif action == "view_attrs" and not self.has_attr():
            return None
        else:
            return True

    def action_exit_dataset_view(self) -> None:
        """Leave full dataset / table / plot view and return focus to the file columns."""
        if isinstance(self.screen, _MODAL_SCREENS_ALL):
            return
        # Plot overlay (including full-screen plot with dataset still open): close plot only.
        if self.has_class("view-plot"):
            self.remove_class("view-plot")
            self._plot_histogram = False
            if not self.has_class("view-dataset"):
                self._data = None
                self._pending_browse_compound_plot = None
                self._pending_browse_histogram_sample = None
                self._compound_browse_histogram_field = None
                self.refresh_preview()
                self._current_list.focus()
            elif self.has_class("view-dtable"):
                self._preview._df.focus()
            else:
                self._preview.focus()
            self.refresh_bindings()
            return
        if self.has_class("view-dataset"):
            # Do not use action_goto_parent(): at file root it no-ops for ``h`` but
            # escape / q must still close the dataset preview.
            self._clear_detail_view_state()
            self._current_list.focus()
            self.update_header(self._header_content(self._cur_dir))
            self.refresh_preview()
            self.refresh_bindings()

    def action_quit_application(self) -> None:
        """Always exit the app (bound to ``Q``)."""
        self.exit()

    def action_quit_contextual(self) -> None:
        """``q``: leave plot or dataset view if open; otherwise quit (file browser only)."""
        if isinstance(self.screen, _MODAL_SCREENS_CHILD):
            scr = self.screen
            if isinstance(scr, SavePlotScreen) and getattr(
                scr, "_histogram_settings_only", False
            ):
                self._on_compound_histogram_browse_cancelled()
            elif isinstance(scr, SavePlotScreen) and getattr(
                scr, "_plot_settings_only", False
            ):
                self._on_compound_plot_settings_cancelled()
            self.pop_screen()
            return
        # Same behavior as Escape (``exit_dataset_view``): close plot overlay first when both
        # ``view-plot`` and ``view-dataset`` are active (e.g. full-screen plot).
        if self.has_class("view-plot") or self.has_class("view-dataset"):
            self.action_exit_dataset_view()
            return
        self.exit()

    def action_view_attrs(self) -> None:
        """Open a modal listing attributes for the selected group or dataset."""
        selected_item = self.get_selected_itemname()
        if selected_item is not None:
            if self.build_attr_str(self._cur_dir, selected_item) != "":
                self.push_screen(
                    AttributeScreen(self._file, self._cur_dir, selected_item)
                )
            else:
                self.notify(
                    "Selected item does not have attributes",
                    severity="warning",
                    timeout=2,
                )

    def _get_1d_save_plot_context(self) -> tuple[np.ndarray | None, str | None]:
        """Return 1D data for save dialog and basename for default ``{{name}}.pdf``."""
        path: str | None = None
        data_arr = None

        if self.has_class("view-dataset"):
            path = self._open_dataset_path
            if getattr(self, "_lazy_mode", False) and self._lazy_dset_path:
                dset = self._file[self._lazy_dset_path]
                if dset.ndim != 1:
                    return None, None
                data_arr = sample_dset_for_plot_1d(dset)
            else:
                data_arr = self._data
        elif self.has_class("view-plot") and self._data is not None:
            path = self.get_selected_dataset_path()
            data_arr = self._data
        else:
            path = self.get_selected_dataset_path()
            if path is None:
                return None, None
            try:
                data_arr = np.asarray(self._file[path][...])
            except Exception:
                return None, None

        if data_arr is None:
            return None, None
        arr = np.asarray(data_arr)
        if arr.dtype.fields is not None:
            if arr.ndim != 1 or not has_compound_plot_fields(arr.dtype):
                return None, None
            base = (path or "plot").rstrip("/")
            name = os.path.basename(base) or "plot"
            return arr, name
        if not is_histogram_eligible(data_arr):
            return None, None
        base = (path or "plot").rstrip("/")
        name = os.path.basename(base) or "plot"
        return arr, name

    def _get_aggregate_data_context(self) -> np.ndarray | None:
        """Numeric array for summary: full dataset view, plot preview, or selected dataset."""
        if self.has_class("view-dataset"):
            if getattr(self, "_lazy_mode", False):
                return None
            d = self._data
            return None if d is None else np.asarray(d)
        if self.has_class("view-plot") and self._data is not None:
            # Browse-mode plot stores downsampled samples in ``_data`` only for drawing.
            return None
        path = self.get_selected_dataset_path()
        if path is None:
            return None
        try:
            return np.asarray(self._file[path][...])
        except Exception:
            return None

    def _preview_aggregate_element_count(self) -> int | None:
        """Element count for summary guard (uses dataset metadata when not yet loaded)."""
        if self.has_class("view-dataset"):
            if getattr(self, "_lazy_mode", False) and self._lazy_dset_path:
                return int(self._file[self._lazy_dset_path].size)
            d = self._data
            return None if d is None else int(np.asarray(d).size)
        path = self.get_selected_dataset_path()
        if path is None:
            return None
        return _dataset_path_element_count(self._file, path)

    def _h5_path_is_plain_numeric_dataset(self, path: str) -> bool:
        """Whether ``path`` is a non-record numeric dataset suitable for summary (no array read)."""
        try:
            node = self._file[path]
        except Exception:
            return False
        if not isinstance(node, h5py.Dataset):
            return False
        if node.dtype.fields is not None:
            return False
        return np.issubdtype(node.dtype, np.number) and node.size > 1

    def _summary_large_confirm_key(self) -> str | None:
        """HDF5 path identifying the dataset for summary double-confirm."""
        if self.has_class("view-dataset"):
            return self._open_dataset_path
        return self.get_selected_dataset_path()

    def _view_dataset_plotable(self) -> bool:
        if getattr(self, "_lazy_mode", False) and self._lazy_dset_path:
            return is_plotable_lazy_dset(self._file[self._lazy_dset_path])
        return self._data is not None and is_plotable(self._data)

    def _view_dataset_element_count(self) -> int:
        if getattr(self, "_lazy_mode", False) and self._lazy_dset_path:
            return int(self._file[self._lazy_dset_path].size)
        if self._data is None:
            return 0
        return int(np.asarray(self._data).size)

    def _view_dataset_histogram_eligible(self) -> bool:
        if getattr(self, "_lazy_mode", False) and self._lazy_dset_path:
            d = self._file[self._lazy_dset_path]
            if d.dtype.fields is not None and d.ndim == 1:
                return has_compound_plot_fields(d.dtype)
            return (
                d.ndim == 1 and len(d.dtype) == 0 and np.issubdtype(d.dtype, np.number)
            )
        if self._data is None:
            return False
        a = np.asarray(self._data)
        if a.dtype.fields is not None and a.ndim == 1:
            return has_compound_plot_fields(a.dtype)
        return is_histogram_eligible(self._data)

    def _compound_histogram_field_from_cursor(self) -> str | None:
        """Numeric field under the DataTable cursor (1D compound full view); else ``None``."""
        if not self.has_class("view-dataset") or not self.has_class("view-dtable"):
            return None
        df = self._preview._df
        if df.row_count == 0 or not df.columns:
            return None
        try:
            c = int(df.cursor_column)
        except (TypeError, ValueError, AttributeError):
            return None
        if c < 0:
            return None

        if getattr(self, "_lazy_mode", False) and getattr(
            self, "_lazy_structured_1d", False
        ):
            lp = self._lazy_dset_path
            if not lp:
                return None
            dset = self._file[lp]
            names = dset.dtype.names
            if not names:
                return None
            nf = len(names)
            fv = min(LAZY_STRUCT_FIELD_WINDOW, nf)
            mx_fs = max(0, nf - fv)
            fs = max(
                0,
                min(int(getattr(self, "_lazy_struct_field_start", 0)), mx_fs),
            )
            nvis = len(df.columns)
            if c >= nvis:
                return None
            gidx = fs + c
            if gidx >= nf:
                return None
            fname = str(names[gidx])
            if fname not in dtype_numeric_field_names(dset.dtype):
                return None
            return fname

        if self._data is None:
            return None
        a = np.asarray(self._data)
        if a.dtype.fields is None or a.ndim != 1:
            return None
        names = a.dtype.names
        if not names:
            return None
        if c >= len(names) or c >= len(df.columns):
            return None
        fname = str(names[c])
        if fname not in dtype_numeric_field_names(a.dtype):
            return None
        return fname

    def _cancel_large_confirm_timer(self) -> None:
        t = self._large_confirm_timer
        if t is not None:
            t.stop()
            self._large_confirm_timer = None

    def _expire_large_confirm_arm(self) -> None:
        """Clear arm when ``set_timer`` fires (see Textual ``MessagePump.set_timer``)."""
        self._large_confirm_timer = None
        self._large_op_arm = None

    def _with_large_dataset_confirm(
        self, arm_key: tuple, n: int, element_threshold: int, run
    ) -> None:
        """Run immediately when ``n`` is at or below ``element_threshold``; else double-press.

        Uses Textual's ``App.notify`` (toast) and ``set_timer`` on the message pump to
        expire the armed state—see ``MessagePump.set_timer`` in the Textual docs.
        """
        if int(n) <= element_threshold:
            self._cancel_large_confirm_timer()
            self._large_op_arm = None
            run()
            return
        if self._large_op_arm == arm_key:
            self._cancel_large_confirm_timer()
            self._large_op_arm = None
            run()
            return
        self._cancel_large_confirm_timer()
        self._large_op_arm = arm_key
        self._large_confirm_timer = self.set_timer(
            _LARGE_CONFIRM_WINDOW_SEC,
            self._expire_large_confirm_arm,
            name="large_dataset_confirm",
        )
        self.notify(
            "Warning: large dataset, may be slow. Press again to confirm.",
            severity="warning",
            timeout=_LARGE_CONFIRM_WINDOW_SEC + 1.0,
        )

    def _start_plot_preview_from_path(self, path: str, *, histogram: bool) -> None:
        """Show plot pane in browse mode using the same downsampling as lazy dataset view."""
        try:
            node = self._file[path]
        except Exception as e:
            self.notify(f"Could not read dataset: {e}", severity="error", timeout=8)
            return
        if not isinstance(node, h5py.Dataset):
            self.notify(f"Not a dataset: {path}", severity="error", timeout=8)
            return
        dset = node

        if histogram:
            if dset.ndim != 1:
                self.notify(
                    "Histogram is only available for 1D datasets",
                    severity="warning",
                    timeout=2,
                )
                return
            if dset.dtype.fields is not None:
                if not has_compound_plot_fields(dset.dtype):
                    self.notify(
                        "Histogram requires numeric compound fields",
                        severity="warning",
                        timeout=2,
                    )
                    return
                try:
                    sampled = sample_dset_for_plot_1d(dset)
                except Exception as e:
                    self.notify(
                        f"Could not read dataset: {e}", severity="error", timeout=8
                    )
                    return
                self._reset_compound_plot_defaults_for_dtype(dset.dtype)
                name = os.path.basename(path.rstrip("/")) or "plot"
                self._pending_browse_histogram_sample = sampled
                self._compound_browse_histogram_field = None
                self.push_screen(
                    SavePlotScreen(
                        sampled,
                        name,
                        compound_x=self._compound_plot_x,
                        compound_y=self._compound_plot_y,
                        compound_log_x=self._compound_plot_log_x,
                        compound_log_y=self._compound_plot_log_y,
                        histogram_settings_only=True,
                    )
                )
                return
            if not (
                len(dset.dtype) == 0
                and dset.dtype != np.dtype("O")
                and np.issubdtype(dset.dtype, np.number)
            ):
                self.notify(
                    "Histogram is only available for 1D numeric datasets",
                    severity="warning",
                    timeout=2,
                )
                return
            try:
                sampled = sample_dset_for_plot_1d(dset)
            except Exception as e:
                self.notify(f"Could not read dataset: {e}", severity="error", timeout=8)
                return
            self._data = sampled
            self._plot_histogram = True
            if not self.has_class("view-plot"):
                self.notify("Histogram…", timeout=1.5)
            # Drop browse snapshot layout or its rules override .view-plot (higher specificity).
            self._preview.remove_class("browse-preview-table")
            self.add_class("view-plot")
            self._preview.replot()
            self._current_list.focus()
            self.refresh_bindings()
            return

        if not is_plotable_lazy_dset(dset):
            self.notify(
                "Only 1D and 2D numeric data can be plotted",
                severity="warning",
                timeout=2,
            )
            return

        try:
            if dset.ndim == 1:
                sampled = sample_dset_for_plot_1d(dset)
            else:
                sampled = sample_dset_for_plot_2d(dset)
        except Exception as e:
            self.notify(f"Could not read dataset: {e}", severity="error", timeout=8)
            return

        if dset.dtype.fields is not None and dset.ndim == 1:
            self._reset_compound_plot_defaults_for_dtype(dset.dtype)
            name = os.path.basename(path.rstrip("/")) or "plot"
            self._pending_browse_compound_plot = (sampled, name)
            self.push_screen(
                SavePlotScreen(
                    sampled,
                    name,
                    compound_x=self._compound_plot_x,
                    compound_y=self._compound_plot_y,
                    compound_log_x=self._compound_plot_log_x,
                    compound_log_y=self._compound_plot_log_y,
                    plot_settings_only=True,
                )
            )
            return
        self._plot_histogram = False
        self._data = sampled
        self.notify("Plotting...", timeout=2)
        self._preview.remove_class("browse-preview-table")
        self.add_class("view-plot")
        self._preview.replot()
        self._current_list.focus()
        self.refresh_bindings()

    def action_save_plot(self) -> None:
        """Open save dialog for 1D numeric or compound data (``s``)."""
        if isinstance(self.screen, _MODAL_SCREENS_ALL):
            return
        data, name = self._get_1d_save_plot_context()
        if data is None or name is None:
            self.notify(
                "Select or open a 1D plottable dataset to save as a plot",
                severity="warning",
                timeout=3,
            )
            return
        if data.dtype.fields is not None and data.ndim == 1:
            self.push_screen(
                SavePlotScreen(
                    data,
                    name,
                    compound_x=self._compound_plot_x,
                    compound_y=self._compound_plot_y,
                    compound_log_x=self._compound_plot_log_x,
                    compound_log_y=self._compound_plot_log_y,
                )
            )
        else:
            self.push_screen(SavePlotScreen(data, name))

    def action_aggregate_data(self) -> None:
        """Open dataset summary modal (``A``): numeric stats in a card grid."""
        if isinstance(self.screen, _MODAL_SCREENS_ALL):
            return
        n = self._preview_aggregate_element_count()
        if n is None:
            self.notify(
                "Select or open a numeric dataset to summarize",
                severity="warning",
                timeout=3,
            )
            return
        if self.has_class("view-dataset"):
            if getattr(self, "_lazy_mode", False) and self._lazy_dset_path:
                if not is_aggregatable_lazy_dset(self._file[self._lazy_dset_path]):
                    self.notify(
                        "Select or open a numeric dataset to summarize",
                        severity="warning",
                        timeout=3,
                    )
                    return
            else:
                data = self._get_aggregate_data_context()
                if data is None or not is_aggregatable(data):
                    self.notify(
                        "Select or open a numeric dataset to summarize",
                        severity="warning",
                        timeout=3,
                    )
                    return
        elif self.has_class("view-plot") and self._data is not None:
            path = self.get_selected_dataset_path()
            if path is None:
                self.notify(
                    "Select or open a numeric dataset to summarize",
                    severity="warning",
                    timeout=3,
                )
                return
            try:
                cand = self._file[path]
            except Exception:
                self.notify(
                    "Select or open a numeric dataset to summarize",
                    severity="warning",
                    timeout=3,
                )
                return
            if not (isinstance(cand, h5py.Dataset) and is_aggregatable_lazy_dset(cand)):
                self.notify(
                    "Select or open a numeric dataset to summarize",
                    severity="warning",
                    timeout=3,
                )
                return
        else:
            path = self.get_selected_dataset_path()
            if path is None or not self._h5_path_is_plain_numeric_dataset(path):
                self.notify(
                    "Select or open a numeric dataset to summarize",
                    severity="warning",
                    timeout=3,
                )
                return

        summary_key = self._summary_large_confirm_key()
        if summary_key is None:
            return

        def _open_summary() -> None:
            if getattr(self, "_lazy_mode", False) and self._lazy_dset_path:
                dset = self._file[self._lazy_dset_path]
                if not is_aggregatable_lazy_dset(dset):
                    return
                try:
                    st = aggregate_h5_numeric_chunks(dset)
                except Exception as e:
                    self.notify(
                        f"Could not summarize dataset: {e}",
                        severity="error",
                        timeout=8,
                    )
                    return
                self.push_screen(DatasetSummaryScreen(streaming_stats=st))
                return
            if self.has_class("view-plot") and not self.has_class("view-dataset"):
                path = self.get_selected_dataset_path()
                if path is None:
                    return
                try:
                    dset = self._file[path]
                except Exception:
                    return
                if not (
                    isinstance(dset, h5py.Dataset) and is_aggregatable_lazy_dset(dset)
                ):
                    return
                try:
                    st = aggregate_h5_numeric_chunks(dset)
                except Exception as e:
                    self.notify(
                        f"Could not summarize dataset: {e}",
                        severity="error",
                        timeout=8,
                    )
                    return
                self.push_screen(DatasetSummaryScreen(streaming_stats=st))
                return
            data = self._get_aggregate_data_context()
            if data is None or not is_aggregatable(data):
                return
            self.push_screen(DatasetSummaryScreen(data))

        self._with_large_dataset_confirm(
            ("summary", summary_key),
            n,
            _LARGE_AGGREGATE_ELEMENT_THRESHOLD,
            _open_summary,
        )

    def action_toggle_dark(self) -> None:
        """Switch between Nord (dark) and Catppuccin Latte (light)."""
        self.theme = "nord" if self.theme == "catppuccin-latte" else "catppuccin-latte"

    def action_goto_parent(self) -> None:
        """Either displays parent or hides dataset"""
        has_parent_dir = self._cur_dir != "/"
        if not has_parent_dir:
            # At file root there is no parent group; do not navigate or clear views
            # (including browse-mode plot in the preview, which uses view-plot without view-dataset).
            return
        if not self.has_class("view-dataset"):
            self._cur_dir = os.path.dirname(self._cur_dir)
            self.update_header(self._header_content(self._cur_dir))
            prev = (
                self._browse_highlight_stack.pop()
                if self._browse_highlight_stack
                else 0
            )
            self.refresh_current_column(prev)
            self.refresh_parent_column()
        self._clear_detail_view_state()
        self._current_list.focus()
        self.update_header(self._header_content(self._cur_dir))
        self.refresh_preview()
        self.refresh_bindings()

    def action_goto_child(self) -> None:
        """Open a group/dataset, or toggle full-width preview when a dataset is open."""
        if self.has_class("view-dataset"):
            # Do not toggle layout while the plot is visible: ``l`` / right would otherwise
            # feel like a random "revert" (columns reappear, focus jumps to the file list).
            if self.has_class("view-plot"):
                return
            # Hide Parent + Current columns so the dataset view uses the full width.
            self.toggle_class("preview-maximized")
            if self.has_class("preview-maximized"):
                if self.has_class("view-dtable"):
                    self._preview._df.focus()
                else:
                    self._preview.focus()
            else:
                self._current_list.focus()
            self.refresh_bindings()
            return
        selected_item = self.get_selected_itemname()
        if selected_item is not None:
            path = h5_child_path(self._cur_dir, selected_item)
            highlighted = self._current_list.highlighted

            blocked = _h5_child_link_blocked_reason(
                self._file, self._cur_dir, selected_item
            )
            if blocked is not None:
                self.notify(blocked, severity="warning", timeout=6)
            else:
                try:
                    node = self._file[path]
                except Exception as e:
                    self.notify(
                        f"Could not open: {e}",
                        severity="warning",
                        timeout=6,
                    )
                else:
                    if isinstance(node, h5py.Group):
                        hi = highlighted if highlighted is not None else 0
                        self._browse_highlight_stack.append(hi)
                        self._cur_dir = path
                        self.update_header(self._header_content(self._cur_dir))
                        self.refresh_current_column(0)
                        self.refresh_parent_column()
                        self.refresh_preview()
                    else:
                        self.update_content(path)
        self.refresh_bindings()

    def action_toggle_plot(self):
        if self.has_class("view-dataset"):
            if not self._view_dataset_plotable():
                self.notify(
                    "Only 1D/2D numeric data or 1D compound (numeric fields) can be plotted",
                    severity="warning",
                )
                return
            opening = not self.has_class("view-plot")
            if opening:
                vk = self._open_dataset_path
                if vk is None:
                    return
                if self._is_compound_dataset_view_for_plot():
                    n = self._view_dataset_element_count()

                    def open_compound_plot_dialog() -> None:
                        sp = self._sample_and_name_compound_plot_dialog()
                        if sp is None:
                            return
                        sampled, name = sp
                        if int(n) > _LARGE_PLOT_ELEMENT_THRESHOLD:
                            self.notify(
                                "Large dataset: points sampled for plot",
                                severity="warning",
                                timeout=4,
                            )
                        self._plot_histogram = False
                        self.push_screen(
                            SavePlotScreen(
                                sampled,
                                name,
                                compound_x=self._compound_plot_x,
                                compound_y=self._compound_plot_y,
                                compound_log_x=self._compound_plot_log_x,
                                compound_log_y=self._compound_plot_log_y,
                                plot_settings_only=True,
                            )
                        )

                    self._with_large_dataset_confirm(
                        (vk, "compound_plot_dialog"),
                        int(n),
                        _LARGE_PLOT_ELEMENT_THRESHOLD,
                        open_compound_plot_dialog,
                    )
                    return
                n = self._view_dataset_element_count()
                if int(n) > _LARGE_PLOT_ELEMENT_THRESHOLD:
                    self.notify(
                        "Large dataset: points sampled for plot",
                        severity="warning",
                        timeout=4,
                    )
                self._plot_histogram = False
                self.notify("Plotting...", timeout=2)
                self.toggle_class("view-plot")
                self._preview.replot()
            else:
                self._plot_histogram = False
                self.toggle_class("view-plot")
                self._preview.replot()
                if not self.has_class("view-plot"):
                    self._plot_histogram = False
                    if self.has_class("view-dtable"):
                        self._preview._df.focus()
                    else:
                        self._preview.focus()
            return

        # Browse mode: plot the selected dataset without opening full preview (``l``).
        if self.has_class("view-plot") and not self.has_class("view-dataset"):
            self.remove_class("view-plot")
            self._plot_histogram = False
            self._data = None
            self._pending_browse_compound_plot = None
            self._pending_browse_histogram_sample = None
            self._compound_browse_histogram_field = None
            self.refresh_preview()
            self._current_list.focus()
            self.refresh_bindings()
            return

        path = self.get_selected_dataset_path()
        if path is None:
            self.notify(
                "Select a dataset in the file list to plot",
                severity="warning",
                timeout=2,
            )
            return
        self._start_plot_preview_from_path(path, histogram=False)

    def action_toggle_plot_histogram(self) -> None:
        """Show a value histogram for 1D numeric data (``P``)."""
        if self.has_class("view-dataset"):
            if not self._view_dataset_histogram_eligible():
                self.notify(
                    "Histogram is only available for 1D numeric datasets",
                    severity="warning",
                    timeout=2,
                )
                return
            if self.has_class("view-plot"):
                self._plot_histogram = True
                self._preview.replot()
                self._preview.focus()
                return
            n = self._view_dataset_element_count()

            vk = self._open_dataset_path
            if vk is None:
                return
            if int(n) > _LARGE_HISTOGRAM_ELEMENT_THRESHOLD:
                self.notify(
                    "Large dataset: points sampled for plot",
                    severity="warning",
                    timeout=4,
                )
            self._plot_histogram = True
            self.notify("Histogram…", timeout=1.5)
            self.add_class("view-plot")
            self._preview.replot()
            self._preview.focus()
            return

        path = self.get_selected_dataset_path()
        if path is None:
            self.notify(
                "Select a 1D numeric dataset in the file list",
                severity="warning",
                timeout=2,
            )
            return
        self._start_plot_preview_from_path(path, histogram=True)


def check_file_validity(fname):
    """Checks if a the provided file is valid"""
    if not fname:
        print("No HDF5 file provided")
        print("Usage: h5tui {file}.h5")
        return False

    if not os.path.isfile(fname):
        print(f"Provided argument '{fname}' is not an existing file.")
        print("Usage: h5tui {file}.h5")
        return False

    if not h5py.is_hdf5(fname):
        print(f"Provided argument '{fname}' is not a valid HDF5 file.")
        print("Usage: h5tui {file}.h5")
        return False

    return True


def h5tui():
    parser = argparse.ArgumentParser(description="H5TUI")
    parser.add_argument("file", type=str, action="store", help="HDF5 file")
    args = parser.parse_args()
    h5file = args.file
    if check_file_validity(h5file):
        H5TUIApp(h5file).run()


if __name__ == "__main__":
    h5tui()
