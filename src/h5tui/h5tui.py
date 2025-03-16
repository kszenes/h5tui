from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, OptionList, Static
from textual.containers import VerticalScroll, Horizontal, Container
from textual.binding import Binding
from textual_plotext import PlotextPlot

import h5py
import numpy as np
import pandas as pd

import sys
import os
import argparse

UNICODE_SUPPORT = sys.stdout.encoding.lower().startswith("utf")


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


class ColumnContent(VerticalScroll):
    """Column which displays a dataset"""

    BINDINGS = [
        Binding("down,j", "scroll_down", "Down", show=True),
        Binding("up,k", "scroll_up", "Up", show=True),
        Binding("u", "page_up", "Bottom", show=False),
        Binding("d", "page_down", "Bottom", show=False),
        Binding("G", "scroll_end", "Bottom", show=False),
        Binding("g", "scroll_home", "Top", show=False),
    ]

    def compose(self):
        self._content = Static(id="data", markup=False)
        self._plot = PlotextPlot(id="plot")
        yield self._content
        yield self._plot

    def update_value(self, value):
        # save value to be able to reference it in toggle truncate
        self._value = value

    def reprint(self):
        """Used to reprint if the numpy formatting is modified"""
        self._content.update(f"{self._value}")

    def replot(self):
        """Plot data, currently only supports 1D and 2D data"""
        data = np.squeeze(self._value)
        if is_plotable(data):
            self._plot.plt.clear_figure()
            if data.ndim == 1:
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


class Column(Container):
    """Column which shows directory structure and selector"""

    def __init__(self, dirs, focus=False):
        super().__init__()
        self._focus = focus
        escaped_dirs = [add_escape_chars(dir) for dir in dirs]
        self._selector_widget = MyOptionList(*escaped_dirs, id="dirs")
        self._content_widget = ColumnContent(id="content")

    def compose(self):
        yield self._selector_widget
        yield self._content_widget
        if self._focus:
            self._selector_widget.focus()

    def update_list(self, dirs, prev_highlighted):
        """Redraw option list with contents of current directory"""
        self._selector_widget.clear_options()
        escaped_dirs = [add_escape_chars(dir) for dir in dirs]
        self._selector_widget.add_options(escaped_dirs)
        self._selector_widget.highlighted = prev_highlighted


class H5TUIApp(App):
    """Simple tui application for displaying and navigating h5 files"""

    BINDINGS = [
        Binding("i", "toggle_dark", "Toggle dark mode", show=False),
        Binding("q", "quit", "Quit", show=False),
        Binding("left,h", "goto_parent", "Parent", show=True, priority=True),
        Binding("right,l", "goto_child", "Select", show=True, priority=True),
        Binding("t", "truncate_print", "Truncate", show=True),
        Binding("s", "suppress_print", "Suppress", show=False),
        Binding("p", "toggle_plot", "Plot", show=True),
        Binding("a", "aggregate_data", "Aggregate", show=True),
    ]
    CSS_PATH = "h5tui.tcss"
    TITLE = "h5tui"

    def __init__(self, fname):
        super().__init__()

        self._fname = fname
        self._file = h5py.File(fname)

        self._cur_dir = str(self._file.name)
        self._dirs = self.get_dir_content(self._cur_dir)

        self._prev_highlighted = 0

        self._truncate_print = True
        self._suppress_print = False
        np.set_printoptions(linewidth=self.size.width)

        self.is_aggregated = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

        self._header_widget = Static("Path: /", id="header", markup=False)
        yield self._header_widget
        with Horizontal():
            dir_with_metadata = self.add_dir_metadata()
            self._column1 = Column(dir_with_metadata, focus=True)
            yield self._column1

    def group_or_dataset(self, elem):
        h5elem = self._file[self._cur_dir + f"/{elem}"]
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

    def add_dir_metadata(self):
        items = list(self._file[self._cur_dir].keys())
        return [self.group_or_dataset(item) + item for item in items]

    def get_dir_content(self, dir) -> list[str]:
        """Return contents of current path"""
        return list(self._file[dir].keys())

    def update_content(self, path):
        self.add_class("view-dataset")

        dset = self._file[path]
        dset_name = os.path.basename(path)
        dset_shape = dset.shape
        dset_dtype = dset.dtype
        dset_data = dset[...]

        self._data = dset_data

        self._column1._content_widget.update_value(self._data)
        self._column1._content_widget.reprint()

        self.update_header(
            f"Path: {self._cur_dir}\nDataset: {dset_name} <{dset_dtype}> {dset_shape}"
        )

    def update_header(self, string):
        self._header_widget.update(string)

    def aggregate_data(self):
        stats = {
            "mean": float(np.mean(self._data)),
            "std": float(np.std(self._data)),
            "max": float(np.max(self._data)),
            "min": float(np.min(self._data)),
            "norm": float(np.linalg.norm(self._data)),
        }

        return stats

    def action_aggregate_data(self):
        if not is_aggregatable(self._data):
            self.notify("Only numeric arrays may be aggregated", severity="warning")
            return

        if not self.is_aggregated:
            content = self._header_widget._content
            stats = self.aggregate_data()
            agg_string = (
                "\nSummary: "
                + "; ".join([f"{key} = {value:.5g}" for key, value in stats.items()])
                + "; "
            )
            self.update_header(content + agg_string)
            self.notify("Summarizing...", timeout=2)
            self.is_aggregated = True

    def action_toggle_dark(self) -> None:
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

    def action_goto_parent(self) -> None:
        """Either displays parent or hides dataset"""
        has_parent_dir = self._cur_dir != "/"
        if has_parent_dir and not self.has_class("view-dataset"):
            self._cur_dir = os.path.dirname(self._cur_dir)
            self._header_widget.update(f"Path: {self._cur_dir}")
            self._column1.update_list(self.add_dir_metadata(), self._prev_highlighted)
        self.is_aggregated = False
        self.remove_class("view-dataset")
        self.remove_class("view-plot")
        # These are the default numpy print setting
        np.set_printoptions(suppress=False, threshold=1000)
        self.update_header(f"Path: {self._cur_dir}")

    def action_goto_child(self) -> None:
        """Either displays child or dataset"""
        highlighted = self._column1._selector_widget.highlighted
        if highlighted is not None:
            selected_item = self._column1._selector_widget.get_option_at_index(
                highlighted
            ).prompt.split()[-1]
            path = os.path.join(self._cur_dir, selected_item)
            path = remove_escaped_chars(path)

            if path in self._file:
                if isinstance(self._file[path], h5py.Group):
                    self._prev_highlighted = highlighted
                    self._cur_dir = path
                    self._header_widget.update(f"Path: {self._cur_dir}")
                    self._column1.update_list(self.add_dir_metadata(), 0)
                else:
                    self.update_content(path)

    def action_truncate_print(self):
        """Change numpy printing by toggling truncation"""
        if self.has_class("view-dataset") and not self.has_class("view-plot"):
            self._truncate_print = not self._truncate_print
            if self._truncate_print:
                default_numpy_truncate = 1000
                np.set_printoptions(threshold=default_numpy_truncate)
                self.notify("Truncation: ON", timeout=2)
            else:
                np.set_printoptions(threshold=sys.maxsize)
                self.notify("Truncation: OFF", timeout=2)
            self._column1._content_widget.reprint()

    def action_suppress_print(self):
        """Change numpy printing by suppression"""
        if self.has_class("view-dataset") and not self.has_class("view-plot"):
            self._suppress_print = not self._suppress_print
            if self._suppress_print:
                np.set_printoptions(suppress=True)
                self.notify("Suppression: ON", timeout=2)
            else:
                np.set_printoptions(suppress=False)
                self.notify("Suppression: OFF", timeout=1)
            self._column1._content_widget.reprint()

    def action_toggle_plot(self):
        if self.has_class("view-dataset"):
            if is_plotable(self._data):
                if not self.has_class("view-plot"):
                    self.notify("Plotting...", timeout=2)
                self.toggle_class("view-plot")
                self._column1._content_widget.replot()
            else:
                self.notify(
                    "Currently only 1D and 2D data is plotable", severity="warning"
                )


def check_file_validity(fname):
    """Checks if a the provided file is valid"""
    if not fname:
        print("No HDF5 file provided")
        print("Usage: h5tui {file}.h5")
        return False

    if not h5py.is_hdf5(fname):
        print(f"Provide argument '{fname}' is not a valid HDF5 file.")
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
