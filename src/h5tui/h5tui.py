from textual import on
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
    Label,
)
from textual.containers import (
    VerticalScroll,
    Horizontal,
    Vertical,
    VerticalGroup,
    HorizontalGroup,
)
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.content import Content
from textual_plotext import PlotextPlot
from rich.markup import escape
from rich.text import Text

import h5py
import numpy as np
import pandas as pd

import sys
import os
import argparse

UNICODE_SUPPORT = sys.stdout.encoding.lower().startswith("utf")

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


def _plot_1d_histogram(plot_plt, data: np.ndarray, bins: int | None = None) -> None:
    """Draw a histogram of ``data`` (1D) using NumPy binning and Plotext ``bar``."""
    vals = np.asarray(data, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        plot_plt.xlabel("Value")
        plot_plt.ylabel("Count")
        return
    nb = _default_histogram_bins(data) if bins is None else max(1, int(bins))
    counts, edges = np.histogram(vals, bins=nb)
    x = ((edges[:-1] + edges[1:]) / 2).tolist()
    y = counts.astype(float).tolist()
    plot_plt.bar(x, y, color="cyan")
    plot_plt.xlabel("Value")
    plot_plt.ylabel("Count")


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


class SavePlotScreen(ModalScreen):
    """Save 1D dataset as PDF: histogram or line plot (matplotlib).

    After a successful save, the file is opened with the OS default app (e.g. Preview).
    """

    BINDINGS = [
        Binding("escape,q", "close_save", "Close", show=True, priority=True),
        Binding("Q", "quit_application", "Quit app", show=False, priority=True),
    ]

    def __init__(self, data: np.ndarray, dset_name: str, id=None) -> None:
        super().__init__(id=id)
        self._data = np.asarray(data)
        self._dset_name = dset_name
        self._default_bins = _default_histogram_bins(self._data)
        safe = dset_name.replace("/", "_").strip() or "plot"
        self._default_filename = f"{safe}.pdf"

    def compose(self) -> ComposeResult:
        with VerticalGroup(id="save-plot-dialog", classes="mode-lineplot"):
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
            with HorizontalGroup(classes="hist-opts"):
                yield Label("Bins:")
                yield Input(
                    value=str(self._default_bins),
                    id="save_bins",
                    placeholder="Bins",
                    type="integer",
                )
            with HorizontalGroup(classes="line-opts"):
                yield Checkbox("Lines", value=True, id="line_lines")
                yield Checkbox("Points", value=False, id="line_points")
                yield Checkbox("Smooth (cubic spline)", value=False, id="line_smooth")
            yield Label("Filename", id="save_filename_label")
            yield Input(value=self._default_filename, id="save_filename")
            yield Button("Save", variant="primary", id="save_plot_btn")
        yield Footer()

    def on_mount(self) -> None:
        self._set_mode_class()
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
        sel = self.query_one("#plot_save_mode", Select)
        return self._mode_str(sel.value)

    def _set_mode_class(self, mode: str | None = None) -> None:
        dlg = self.query_one("#save-plot-dialog", VerticalGroup)
        if mode is None:
            mode = self._mode()
        dlg.set_classes(f"mode-{mode}")

    @on(Select.Changed, "#plot_save_mode")
    def _on_mode_changed(self, event: Select.Changed) -> None:
        # Use the message value: ``Select.value`` can still be the previous value
        # during this handler on some Textual versions / focus paths.
        mode = self._mode_str(event.value)
        self._set_mode_class(mode)

    def action_close_save(self) -> None:
        self.app.pop_screen()

    def action_quit_application(self) -> None:
        self.app.exit()

    @on(Button.Pressed, "#save_plot_btn")
    def _on_save_pressed(self) -> None:
        path = self.query_one("#save_filename", Input).value.strip()
        if not path:
            self.app.notify("Enter a filename", severity="warning", timeout=2)
            return
        mode = self._mode()
        try:
            if mode == "histogram":
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

    def __init__(self, data: np.ndarray, id=None) -> None:
        super().__init__(id=id)
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


def _cell_value_to_str(value: object) -> str:
    """Plain string for DataTable cell contents (clipboard / display)."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Text):
        return value.plain
    return str(value)


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
        try:
            value = self.get_cell_at(self.cursor_coordinate)
        except Exception:
            return
        self.app.copy_to_clipboard(_cell_value_to_str(value))
        self.app.notify("Copied to clipboard", timeout=1.5)

    def update(self, value):
        self.clear(columns=True)
        self.add_columns(*get_colnames(value))
        for row in value:
            row_cleaned = [
                e.decode("utf8") if isinstance(e, bytes) else e for e in row.item()
            ]
            self.add_row(*row_cleaned)

    def update_1d(self, arr: np.ndarray) -> None:
        """Fill one unnamed column; row labels are ``0``, ``1``, … (no colon; text preview uses ``i:``)."""
        a = np.asarray(arr)
        self.clear(columns=True)
        self.add_columns("")
        if a.size == 0:
            return
        for i in range(int(a.shape[0])):
            self.add_row(
                _format_1d_value_display(a[i]),
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


def get_colnames(obj):
    """Return the column names of numpy recarray"""
    return obj.dtype.names


def is_dataframe(obj):
    """Checks if numpy array is a dataframe i.e., recarray"""
    return len(obj.dtype) != 0


def is_plotable(obj):
    is_not_composity_type = len(obj.dtype) == 0
    squeezed = np.squeeze(obj)
    return is_not_composity_type and (squeezed.ndim == 1 or squeezed.ndim == 2)


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
    """Non-record 1D arrays we show as one line per index (exclude object arrays)."""
    if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        return False
    return arr.dtype.fields is None and arr.dtype != np.dtype("O")


# Plain ndarray DataTable limits (text fallback beyond these).
MAX_TABLE_ELEMENTS = 200_000
MAX_TABLE_COLUMNS = 1024


def should_use_1d_table(arr) -> bool:
    """Use a labeled-row DataTable (index + value column) for plain 1D data."""
    a = _as_display_1d(np.asarray(arr))
    return is_plain_1d_array(a) and a.size <= MAX_TABLE_ELEMENTS


def is_plain_2d_array(arr: np.ndarray) -> bool:
    """Non-record 2D arrays (exclude object dtype)."""
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        return False
    return arr.dtype.fields is None and arr.dtype != np.dtype("O")


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


def _format_array_element(v) -> str:
    x = np.asarray(v)
    return np.array2string(
        x,
        max_line_width=10_000,
        suppress_small=False,
        separator=", ",
    )


def _format_table_cell(v) -> str:
    """Single-cell text for DataTable rows.

    Integers and bools use a fast path. Real/complex scalars go through
    ``array2string`` for consistent NumPy-style text output.
    """
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
    if np.issubdtype(x.dtype, np.bool):
        return str(bool(e))
    if np.issubdtype(x.dtype, np.integer):
        return str(int(e))
    if np.issubdtype(x.dtype, np.inexact):
        return _format_array_element(v)
    return _format_array_element(v)


def _format_1d_value_display(v: object) -> str:
    """Format one scalar for 1D table/preview: space before positive signs (column alignment)."""
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
        return f"{i:>{idxw}}: {_format_1d_value_display(a[i])}"

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


def add_escape_chars(string: str):
    return string.replace("[", r"\[")


def remove_escaped_chars(string: str):
    return string.replace(r"\[", "[")


class MyOptionList(OptionList):
    BINDINGS = [
        Binding("down,j", "cursor_down", "Down", show=True),
        Binding("up,k", "cursor_up", "Up", show=True),
        Binding("G", "page_down", "Bottom", show=False),
        Binding("g", "page_up", "Top", show=False),
    ]

    def action_cursor_down(self) -> None:
        self.refresh_bindings()
        return super().action_cursor_down()

    def action_cursor_up(self) -> None:
        self.refresh_bindings()
        return super().action_cursor_up()

    def check_action(self, action, parameters):
        if action in ["cursor_down", "cursor_up"] and self.app.has_class(
            "view-dataset"
        ):
            return False
        else:
            return True


class ColumnContent(VerticalScroll):
    """Column which displays a dataset"""

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
        self._df = MyDataTable(id="dtable")
        yield self._content
        yield self._plot
        yield self._df

    def reprint(self):
        """Refresh dataset text or table content using ``H5TUIApp._data``."""
        app = self.app
        data = getattr(app, "_data", None)
        if data is None:
            return
        lw = int(getattr(app, "_preview_line_width", _ND_LINEWIDTH_DEFAULT))
        if is_dataframe(data):
            self.notify("keys: h/j/k/l; H/J/K/L jump ±10; y copy; esc quit")
            self._df.update(data)
            self._df.focus()
        elif should_use_1d_table(data):
            self.notify("keys: j/k row; J/K jump ±10; y copy; esc quit")
            self._df.update_1d(_as_display_1d(np.asarray(data)))
            self._df.focus()
        elif should_use_2d_table(data):
            self.notify("keys: h/j/k/l; H/J/K/L jump ±10; y copy; esc quit")
            self._df.update_2d(np.asarray(data))
            self._df.focus()
        else:
            self._content.update(format_ndarray_text(np.asarray(data), line_width=lw))

    def replot(self):
        """Plot data, currently only supports 1D and 2D data"""
        raw = getattr(self.app, "_data", None)
        if raw is None:
            return
        data = np.squeeze(raw)
        if is_plotable(data):
            self._plot.plt.clear_figure()
            if data.ndim == 1 and getattr(self.app, "_plot_histogram", False):
                _plot_1d_histogram(self._plot.plt, data)
            elif data.ndim == 1:
                self._plot.plt.xlabel("Index")
                self._plot.plt.plot(
                    np.arange(data.shape[0]),
                    data,
                    color="cyan",
                    marker="braille",
                )
            elif data.ndim == 2:
                nrows, ncols = data.shape
                self._plot.plt.plot_size(nrows, ncols)
                # arbitrary, should be expermineted with
                size_threshold = 100
                if nrows < size_threshold and ncols < size_threshold:
                    self._plot.plt.heatmap(pd.DataFrame(data))
                    # heatmap has default title, remove it
                    self._plot.plt.title("")
                else:
                    self._plot.plt.matrix_plot(data.tolist())
                self._plot.plt.xlabel("Column")
                self._plot.plt.ylabel("Row")


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

        self._cur_dir = str(self._file.name)

        self._prev_highlighted = 0
        self._preview_line_width = _ND_LINEWIDTH_DEFAULT
        self._data = None

        self._refresh_guard = False
        # When True, 1D plot view shows a value histogram instead of index vs value.
        self._plot_histogram = False
        # HDF5 path for the dataset opened in full view (basename used for save-as default).
        self._open_dataset_path: str | None = None

    def on_unmount(self) -> None:
        self._file.close()

    def _header_content(
        self, h5_path: str, *, dataset_meta: str | None = None
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

    def group_or_dataset(self, dir_path: str, elem: str):
        path = h5_child_path(dir_path, elem)
        h5elem = self._file[path]
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

    def metadata_line_for_item(self, dir_path: str, item: str) -> str:
        return (
            self.group_or_dataset(dir_path, item)
            + item
            + f"    {self.build_attr_str(dir_path, item)}"
        )

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
        h5elem = self._file[path]
        num_attrs = len(h5elem.attrs)
        if num_attrs > 0:
            return f"▼ ({num_attrs})"
        else:
            return ""

    def add_dir_metadata_for(self, dir_path: str):
        items = list(self._file[dir_path].keys())
        return [self.metadata_line_for_item(dir_path, item) for item in items]

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

    def _dataset_preview_block(self, dset: h5py.Dataset) -> str:
        lw = int(self._preview_line_width)
        lines = [f"shape: {dset.shape}", f"dtype: {dset.dtype}\n"]
        if dset.size == 0:
            lines.append("(empty)")
            return "\n".join(lines)
        if len(dset.dtype) != 0:
            lines.append("(compound dtype — press l for table view)")
            return "\n".join(lines)
        try:
            if dset.size <= 128:
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
                lines.append("(large array — press l for full view)")
        except Exception as e:
            lines.append(f"(could not read data: {e})")
        return "\n".join(lines)

    def _clear_detail_view_state(self) -> None:
        """Leave full dataset / plot / table view and restore default printing."""
        self._data = None
        self._open_dataset_path = None
        self._plot_histogram = False
        self.remove_class("view-dataset")
        self.remove_class("view-plot")
        self.remove_class("view-dtable")
        self.remove_class("preview-maximized")

    def refresh_preview(self) -> None:
        """Update the preview pane from the Current column selection (browse mode)."""
        if self.has_class("view-dataset"):
            return
        if self.has_class("view-plot") and not self.has_class("view-dataset"):
            self.remove_class("view-plot")
            self._plot_histogram = False
            self._data = None
        self._preview.remove_class("empty-group-preview")
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
                lines = [self.metadata_line_for_item(path, k) for k in shown]
                text = "\n".join(lines)
                if more > 0:
                    text += f"\n… ({more} more)"
                self._preview._content.update(text)
        elif isinstance(node, h5py.Dataset):
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
            self._prev_highlighted = 0
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

    def update_content(self, path):
        self._preview.remove_class("empty-group-preview")
        self.remove_class("view-plot")
        self._plot_histogram = False
        try:
            dset = self._file[path]
            dset_shape = dset.shape
            dset_data = dset[...]
        except Exception as e:
            self.notify(f"Could not read dataset: {e}", severity="error", timeout=8)
            return

        self._data = dset_data
        self._open_dataset_path = path

        self.add_class("view-dataset")
        self.add_class("preview-maximized")
        if is_dataframe(self._data):
            self.add_class("view-dtable")
            dset_dtype = [str(field[0]) for field in dset.dtype.fields.values()]
        else:
            dset_dtype = dset.dtype
            if should_use_ndarray_datatable(self._data):
                self.add_class("view-dtable")

        self._preview.reprint()

        self.update_header(
            self._header_content(
                path,
                dataset_meta=f"<{dset_dtype}> {dset_shape}",
            )
        )
        # Focus DataTable when it is the active view; focusing the parent ColumnContent
        # would steal focus from the table (requires mouse click to use j/k on rows).
        if self.has_class("view-dtable"):
            self._preview._df.focus()
        else:
            self._preview.focus()

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
            self.screen, (AttributeScreen, SavePlotScreen, DatasetSummaryScreen)
        ):
            # Attribute/save/summary modals use J/K for inner scroll.
            return False
        elif action in ("scroll_preview_down", "scroll_preview_up") and (
            self.focused is self._preview._df
        ):
            # DataTable owns J/K for ±10 row jump; do not run app handler.
            return False
        elif action in ["cursor_down", "cursor_up"] and self.has_class("view-dataset"):
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
        if isinstance(
            self.screen, (AttributeScreen, SavePlotScreen, DatasetSummaryScreen)
        ):
            return
        if self.has_class("view-plot") and not self.has_class("view-dataset"):
            self.remove_class("view-plot")
            self._plot_histogram = False
            self._data = None
            self.refresh_preview()
            self._current_list.focus()
            self.refresh_bindings()
            return
        if self.has_class("view-dataset"):
            self.action_goto_parent()

    def action_quit_application(self) -> None:
        """Always exit the app (bound to ``Q``)."""
        self.exit()

    def action_quit_contextual(self) -> None:
        """``q``: leave plot or dataset view if open; otherwise quit (file browser only)."""
        if isinstance(self.screen, (SavePlotScreen, DatasetSummaryScreen)):
            self.pop_screen()
            return
        if self.has_class("view-plot"):
            self.remove_class("view-plot")
            self._plot_histogram = False
            if not self.has_class("view-dataset"):
                self._data = None
                self.refresh_preview()
                self._current_list.focus()
            elif self.has_class("view-dtable"):
                self._preview._df.focus()
            else:
                self._preview.focus()
            self.refresh_bindings()
            return
        if self.has_class("view-dataset"):
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
        """Return 1D numeric data and a short name for default ``{name}.pdf``, or ``(None, None)``."""
        path: str | None = None
        data_arr = None

        if self.has_class("view-dataset"):
            path = self._open_dataset_path
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
        if not is_histogram_eligible(data_arr):
            return None, None
        base = (path or "plot").rstrip("/")
        name = os.path.basename(base) or "plot"
        return np.asarray(data_arr), name

    def _get_aggregate_data_context(self) -> np.ndarray | None:
        """Numeric array for summary: full dataset view, plot preview, or selected dataset."""
        if self.has_class("view-dataset"):
            d = self._data
            return None if d is None else np.asarray(d)
        if self.has_class("view-plot") and self._data is not None:
            return np.asarray(self._data)
        path = self.get_selected_dataset_path()
        if path is None:
            return None
        try:
            return np.asarray(self._file[path][...])
        except Exception:
            return None

    def action_save_plot(self) -> None:
        """Open save dialog for 1D numeric data (``s``)."""
        if isinstance(
            self.screen, (AttributeScreen, SavePlotScreen, DatasetSummaryScreen)
        ):
            return
        data, name = self._get_1d_save_plot_context()
        if data is None or name is None:
            self.notify(
                "Select or open a 1D numeric dataset to save as a plot",
                severity="warning",
                timeout=3,
            )
            return
        self.push_screen(SavePlotScreen(data, name))

    def action_aggregate_data(self) -> None:
        """Open dataset summary modal (``A``): numeric stats in a card grid."""
        if isinstance(
            self.screen, (AttributeScreen, SavePlotScreen, DatasetSummaryScreen)
        ):
            return
        data = self._get_aggregate_data_context()
        if data is None or not is_aggregatable(data):
            self.notify(
                "Select or open a numeric dataset to summarize",
                severity="warning",
                timeout=3,
            )
            return
        self.push_screen(DatasetSummaryScreen(data))

    def action_toggle_dark(self) -> None:
        """Switch between Nord (dark) and Catppuccin Latte (light)."""
        self.theme = "nord" if self.theme == "catppuccin-latte" else "catppuccin-latte"

    def action_goto_parent(self) -> None:
        """Either displays parent or hides dataset"""
        has_parent_dir = self._cur_dir != "/"
        if has_parent_dir and not self.has_class("view-dataset"):
            self._cur_dir = os.path.dirname(self._cur_dir)
            self.update_header(self._header_content(self._cur_dir))
            self.refresh_current_column(self._prev_highlighted)
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

            if path in self._file:
                if isinstance(self._file[path], h5py.Group):
                    self._prev_highlighted = highlighted
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
            if is_plotable(self._data):
                opening = not self.has_class("view-plot")
                if opening:
                    self._plot_histogram = False
                    self.notify("Plotting...", timeout=2)
                self.toggle_class("view-plot")
                self._preview.replot()
                if not self.has_class("view-plot"):
                    self._plot_histogram = False
                    if self.has_class("view-dtable"):
                        self._preview._df.focus()
                    else:
                        self._preview.focus()
            else:
                self.notify(
                    "Currently only 1D and 2D data is plotable", severity="warning"
                )
            return

        # Browse mode: plot the selected dataset without opening full preview (``l``).
        if self.has_class("view-plot") and not self.has_class("view-dataset"):
            self.remove_class("view-plot")
            self._plot_histogram = False
            self._data = None
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
        try:
            dset_data = np.asarray(self._file[path][...])
        except Exception as e:
            self.notify(f"Could not read dataset: {e}", severity="error", timeout=8)
            return
        if not is_plotable(dset_data):
            self.notify(
                "Only 1D and 2D numeric data can be plotted",
                severity="warning",
                timeout=2,
            )
            return
        self._plot_histogram = False
        self._data = dset_data
        self.notify("Plotting...", timeout=2)
        self.add_class("view-plot")
        self._preview.replot()
        # Keep focus on the file list so j/k and arrows still change selection.
        self._current_list.focus()
        self.refresh_bindings()

    def action_toggle_plot_histogram(self) -> None:
        """Show a value histogram for 1D numeric data (``P``)."""
        if self.has_class("view-dataset"):
            if self._data is None or not is_histogram_eligible(self._data):
                self.notify(
                    "Histogram is only available for 1D numeric datasets",
                    severity="warning",
                    timeout=2,
                )
                return
            self._plot_histogram = True
            if not self.has_class("view-plot"):
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
        try:
            dset_data = np.asarray(self._file[path][...])
        except Exception as e:
            self.notify(f"Could not read dataset: {e}", severity="error", timeout=8)
            return
        self._data = dset_data
        if not is_histogram_eligible(self._data):
            self._data = None
            self.notify(
                "Histogram is only available for 1D numeric datasets",
                severity="warning",
                timeout=2,
            )
            return
        self._plot_histogram = True
        if not self.has_class("view-plot"):
            self.notify("Histogram…", timeout=1.5)
        self.add_class("view-plot")
        self._preview.replot()
        self._current_list.focus()
        self.refresh_bindings()


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
