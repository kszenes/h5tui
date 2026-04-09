from textual.app import App, ComposeResult
from textual.reactive import Reactive
from textual.widgets import Footer, Header, OptionList, Static, DataTable
from textual.containers import VerticalScroll, Horizontal, Vertical
from textual.binding import Binding
from textual.screen import ModalScreen
from textual_plotext import PlotextPlot

import h5py
import numpy as np
import pandas as pd

import sys
import os
import argparse

UNICODE_SUPPORT = sys.stdout.encoding.lower().startswith("utf")


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
            "left,h,q",
            "quit_attrs",
            "Return",
            show=True,
            priority=True,
        ),
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


class MyDataTable(DataTable):
    BINDINGS = [
        Binding("enter", "select_cursor", "Select", show=False),
        Binding("up,k", "cursor_up", "Cursor up", show=False),
        Binding("down,j", "cursor_down", "Cursor down", show=False),
        Binding("right,L", "cursor_right", "Cursor right", show=False),
        Binding("left,H", "cursor_left", "Cursor left", show=False),
        Binding("pageup,u", "page_up", "Page up", show=False),
        Binding("pagedown,d", "page_down", "Page down", show=False),
        Binding("g", "scroll_top", "Top", show=False),
        Binding("G", "scroll_bottom", "Bottom", show=False),
    ]

    def __init__(self, id):
        super().__init__(id=id)

    def update(self, value):
        self.clear(columns=True)
        self.add_columns(*get_colnames(value))
        for row in value:
            row_cleaned = [
                e.decode("utf8") if isinstance(e, bytes) else e for e in row.item()
            ]
            self.add_row(*row_cleaned)


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

    def update_value(self, value):
        # save value to be able to reference it in toggle truncate
        self._value = value

    def reprint(self):
        """Used to reprint if the numpy formatting is modified"""
        if is_dataframe(self._value):
            self.notify("Entering data table: use capital H and L to navigate columns")
            self._df.update(self._value)
            self._df.focus()

        else:
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


class H5TUIApp(App):
    """Simple tui application for displaying and navigating h5 files"""

    theme = Reactive("nord")

    BINDINGS = [
        Binding("i", "toggle_dark", "Toggle dark mode", show=False),
        Binding("q", "quit", "Quit", show=False),
        Binding("left,h", "goto_parent", "Back", show=True),
        Binding("right,l", "goto_child", "Select", show=True, priority=True),
        Binding("a", "view_attrs", "Attributes", show=True),
        Binding("t", "truncate_print", "Truncate", show=True),
        Binding("s", "suppress_print", "Suppress", show=True),
        Binding("p", "toggle_plot", "Plot", show=True),
        Binding("A", "aggregate_data", "Aggregate", show=True),
    ]
    CSS_PATH = "h5tui.tcss"
    TITLE = "h5tui"

    def __init__(self, fname):
        super().__init__()

        self._fname = fname
        self._file = h5py.File(fname)

        self._cur_dir = str(self._file.name)

        self._prev_highlighted = 0

        self._truncate_print = True
        self._suppress_print = False
        np.set_printoptions(linewidth=self.size.width)

        self.is_aggregated = False
        self._refresh_guard = False
        # Full header text while a dataset is open (used for aggregate; avoids broken
        # concatenation from Static.content / non-string renderables).
        self._dataset_detail_header = ""

    def on_unmount(self) -> None:
        self._file.close()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

        self._header_widget = Static("Path: /", id="header", markup=False)
        yield self._header_widget
        with Horizontal(id="browser"):
            self._parent_list = MyOptionList(id="dirs-parent", markup=False)
            self._current_list = MyOptionList(id="dirs-current", markup=False)
            self._preview = ColumnContent(id="content")
            yield self._parent_list
            yield self._current_list
            yield self._preview

    def on_mount(self) -> None:
        self._parent_list.border_title = "Parent"
        self._current_list.border_title = "Current"
        self._preview.border_title = "Preview"
        self.refresh_parent_column()
        self.refresh_current_column(0)
        self.refresh_preview()
        self._current_list.focus()

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
        name = os.path.basename(dset.name.rstrip("/")) or dset.name
        lines = [name, f"shape: {dset.shape}", f"dtype: {dset.dtype}"]
        if dset.size == 0:
            lines.append("(empty)")
            return "\n".join(lines)
        if len(dset.dtype) != 0:
            lines.append("(compound dtype — press l for table view)")
            return "\n".join(lines)
        if dset.size <= 128:
            arr = np.asarray(dset[...])
            lines.append(np.array2string(arr, threshold=100, edgeitems=3))
            return "\n".join(lines)
        if dset.ndim == 1:
            n = min(24, dset.shape[0])
            lines.append(np.array2string(np.asarray(dset[:n]), threshold=n + 1))
            lines.append("… (press l for full view)")
        elif dset.ndim == 2:
            r = min(6, dset.shape[0])
            c = min(10, dset.shape[1])
            lines.append(
                np.array2string(np.asarray(dset[:r, :c]), threshold=200, edgeitems=2)
            )
            lines.append("… (press l for full view)")
        else:
            lines.append("(large array — press l for full view)")
        return "\n".join(lines)

    def _clear_detail_view_state(self) -> None:
        """Leave full dataset / plot / table view and restore default printing."""
        self.is_aggregated = False
        self._dataset_detail_header = ""
        self.remove_class("view-dataset")
        self.remove_class("view-plot")
        self.remove_class("view-dtable")
        self.remove_class("preview-maximized")
        np.set_printoptions(suppress=False, threshold=1000)

    def refresh_preview(self) -> None:
        """Update the preview pane from the Current column selection (browse mode)."""
        if self.has_class("view-dataset"):
            return
        name = self.get_selected_itemname()
        if name is None:
            self._preview._content.update("(empty directory)")
            return
        path = h5_child_path(self._cur_dir, name)
        node = self._file[path]
        if isinstance(node, h5py.Group):
            keys = list(node.keys())
            if not keys:
                text = "(empty group)"
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
            self.update_header(f"Path: {self._cur_dir}")
            self.refresh_bindings()
            return
        if oid == "dirs-current":
            # Mouse (or other) moves the highlight while a dataset is open: leave
            # full view so the preview matches the new selection.
            if self.has_class("view-dataset"):
                self._clear_detail_view_state()
            self.refresh_preview()

    def update_content(self, path):
        dset = self._file[path]
        dset_name = os.path.basename(path)
        dset_shape = dset.shape
        dset_data = dset[...]

        self._data = dset_data

        self.add_class("view-dataset")
        self.add_class("preview-maximized")
        if is_dataframe(self._data):
            self.add_class("view-dtable")
            dset_dtype = [str(field[0]) for field in dset.dtype.fields.values()]
        else:
            dset_dtype = dset.dtype

        self._preview.update_value(self._data)
        self._preview.reprint()

        self._dataset_detail_header = (
            f"Path: {self._cur_dir}\nDataset: {dset_name} <{dset_dtype}> {dset_shape}"
        )
        self.update_header(self._dataset_detail_header)
        self._preview.focus()

    def update_header(self, string):
        self._header_widget.update(string)

    def aggregate_data(self):
        stats = {
            "mean": float(np.mean(self._data)),
            "std": float(np.std(self._data)),
            "max": float(np.max(self._data)),
            "min": float(np.min(self._data)),
            "L2 norm": float(np.linalg.norm(self._data)),
        }

        return stats

    def check_action(self, action, parameters):
        if (
            action
            in [
                "truncate_print",
                "suppress_print",
                "toggle_plot",
                "aggregate_data",
            ]
        ) and not self.has_class("view-dataset"):
            return False
        elif action in ["cursor_down", "cursor_up"] and self.has_class("view-dataset"):
            return False
        elif action == "view_attrs" and not self.has_attr():
            return None
        else:
            return True

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

    def action_aggregate_data(self):
        if self.has_class("view-dataset"):
            if not is_aggregatable(self._data):
                self.notify(
                    "Only numeric arrays may be aggregated",
                    severity="warning",
                    timeout=2,
                )
                return

            if not self.is_aggregated:
                base = self._dataset_detail_header or f"Path: {self._cur_dir}"
                stats = self.aggregate_data()
                agg_string = (
                    "\nSummary: "
                    + "; ".join(
                        [f"{key} = {value:.5g}" for key, value in stats.items()]
                    )
                    + "; "
                )
                self._dataset_detail_header = base + agg_string
                self.update_header(self._dataset_detail_header)
                self.notify("Summarizing...", timeout=2)
                self.is_aggregated = True

    def action_toggle_dark(self) -> None:
        """Switch between Nord (dark) and Textual light theme."""
        self.theme = "textual-light" if self.theme == "nord" else "nord"

    def action_goto_parent(self) -> None:
        """Either displays parent or hides dataset"""
        has_parent_dir = self._cur_dir != "/"
        if has_parent_dir and not self.has_class("view-dataset"):
            self._cur_dir = os.path.dirname(self._cur_dir)
            self._header_widget.update(f"Path: {self._cur_dir}")
            self.refresh_current_column(self._prev_highlighted)
            self.refresh_parent_column()
        self._clear_detail_view_state()
        self._current_list.focus()
        self.update_header(f"Path: {self._cur_dir}")
        self.refresh_preview()
        self.refresh_bindings()

    def action_goto_child(self) -> None:
        """Open a group/dataset, or toggle full-width preview when a dataset is open."""
        if self.has_class("view-dataset"):
            # Hide Parent + Current columns so the dataset view uses the full width.
            self.toggle_class("preview-maximized")
            if self.has_class("preview-maximized"):
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
                    self._header_widget.update(f"Path: {self._cur_dir}")
                    self.refresh_current_column(0)
                    self.refresh_parent_column()
                    self.refresh_preview()
                else:
                    self.update_content(path)
        self.refresh_bindings()

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
            self._preview.reprint()

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
            self._preview.reprint()

    def action_toggle_plot(self):
        if self.has_class("view-dataset"):
            if is_plotable(self._data):
                if not self.has_class("view-plot"):
                    self.notify("Plotting...", timeout=2)
                self.toggle_class("view-plot")
                self._preview.replot()
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
