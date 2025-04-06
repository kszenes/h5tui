from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, OptionList, Static, DataTable
from textual.containers import VerticalScroll, Horizontal, Container, Vertical
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
        Binding("u", "scroll_content_page_up", "Scroll Down", priority=True),
        Binding("d", "scroll_content_page_down", "Scroll Up", priority=True),
    ]

    def __init__(self, h5file, cur_dir, itemname, id=None) -> None:
        super().__init__(id=id)
        self._file = h5file
        self._cur_dir = cur_dir
        self._itemname = itemname
        self._item = self._file[self._cur_dir + f"/{self._itemname}"]
        self._attrs = list(self._item.attrs.keys())

        self._cur_attr = self._attrs[0]

        self._selector_widget = MyOptionList(*self._attrs)
        self._selector_widget.border_title = f"Attributes for {self._itemname}"

        self._content_widget = Static(id="attr_content")
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
        print(f"{content = }")
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
        print("scrolling down")
        self._vertical_widget.scroll_down()

    def action_scroll_content_up(self):
        print("scrolling up")
        self._vertical_widget.scroll_up()

    def action_scroll_content_page_down(self):
        print("scrolling DOWN")
        self._vertical_widget.scroll_page_down()

    def action_scroll_content_page_up(self):
        print("scrolling UP")
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

    def has_attr(self):
        """Return if the currently selected item has attributes"""
        highlighted = self._column1._selector_widget.highlighted
        if highlighted is not None:
            prompt = self._column1._selector_widget.get_option_at_index(
                highlighted
            ).prompt
            selected_item = remove_escaped_chars(self.get_itemname_from_prompt(prompt))
            return self.build_attr_str(selected_item) != ""

    def build_attr_str(self, elem):
        """Creates the has attributes string (▼ + num attrs)"""
        h5elem = self._file[self._cur_dir + f"/{elem}"]
        num_attrs = len(h5elem.attrs)
        if num_attrs > 0:
            return f"▼ ({num_attrs})"
        else:
            return ""

    def add_dir_metadata(self):
        items = list(self._file[self._cur_dir].keys())
        with_type_icon = [self.group_or_dataset(item) + item for item in items]
        with_attrs = [
            with_type + f"    {self.build_attr_str(item)}"
            for with_type, item in zip(with_type_icon, items)
        ]
        return with_attrs

    def get_itemname_from_prompt(self, prompt):
        """
        Returns the item name from the selected item
        The selected item contains an icon for group or dset, itemname and the number of attributes
        """
        return prompt.split()[1]

    def get_dir_content(self, dir):
        """Return contents of current path"""
        return list(self._file[dir].keys())

    def update_content(self, path):
        dset = self._file[path]
        dset_name = os.path.basename(path)
        dset_shape = dset.shape
        dset_data = dset[...]

        self._data = dset_data

        self.add_class("view-dataset")
        if is_dataframe(self._data):
            self.add_class("view-dtable")
            dset_dtype = [str(field[0]) for field in dset.dtype.fields.values()]
        else:
            dset_dtype = dset.dtype

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
        elif action == "goto_child" and self.has_class("view-dataset"):
            return False
        elif action in ["cursor_down", "cursor_up"] and self.has_class("view-dataset"):
            return False
        elif action == "view_attrs" and not self.has_attr():
            return None
        else:
            return True

    def action_view_attrs(self) -> None:
        """Action to display the quit dialog."""
        highlighted = self._column1._selector_widget.highlighted
        if highlighted is not None:
            prompt = self._column1._selector_widget.get_option_at_index(
                highlighted
            ).prompt
            selected_item = remove_escaped_chars(self.get_itemname_from_prompt(prompt))
            if self.build_attr_str(selected_item) != "":
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
                content = self._header_widget._content
                stats = self.aggregate_data()
                agg_string = (
                    "\nSummary: "
                    + "; ".join(
                        [f"{key} = {value:.5g}" for key, value in stats.items()]
                    )
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
        self.remove_class("view-dtable")
        self._column1._selector_widget.focus()
        # These are the default numpy print setting
        np.set_printoptions(suppress=False, threshold=1000)
        self.update_header(f"Path: {self._cur_dir}")
        self.refresh_bindings()

    def action_goto_child(self) -> None:
        """Either displays child or dataset"""
        if (
            self.has_class("view-dataset")
            or self.has_class("view-plot")
            or self.has_class("view-dtable")
        ):
            # Does not do anything if data is already being viewed
            return
        highlighted = self._column1._selector_widget.highlighted
        if highlighted is not None:
            prompt = self._column1._selector_widget.get_option_at_index(
                highlighted
            ).prompt
            selected_item = self.get_itemname_from_prompt(prompt)
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
