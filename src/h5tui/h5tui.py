from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, OptionList, Static
from textual.containers import VerticalScroll, Horizontal, Container
from textual.binding import Binding

import h5py
import numpy as np

import sys
import os
import argparse


class MyOptionList(OptionList):
    BINDINGS = [
        Binding("down,j", "cursor_down", "Down", show=True),
        Binding("up,k", "cursor_up", "Up", show=True),
        Binding("G", "page_down", "Bottom", show=False),
        Binding("g", "page_up", "Top", show=False),
    ]


# TODO: This shoudln't be a VerticalScroll, just a widget and the
#       content should be a VerticalScroll, breaks the h keybinding for some reason
class ColumnContent(VerticalScroll):
    """Column which displays a dataset"""

    BINDINGS = [
        Binding("down,j", "scroll_down", "Down", show=True),
        Binding("up,k", "scroll_up", "Up", show=True),
        Binding("G", "scroll_end", "Bottom", show=False),
        Binding("g", "scroll_home", "Top", show=False),
        Binding("t", "truncate_print", "Truncate print", show=False),
    ]

    def __init__(self, id):
        super().__init__(id=id)
        self.truncate_print = True

    def compose(self):
        self._header = Static(classes="header")
        self._content = Static()

        yield self._header
        yield self._content

    def update(self, path, name, shape, value):
        # save value to be able to call it in toggle truncate
        self._value = value

        self._header.update(f"Path: {path}\nDataset: {name} {shape}")
        self._content.update(f"{self._value}")

    def action_truncate_print(self):
        self.truncate_print = not self.truncate_print
        if self.truncate_print:
            default_numpy_truncate = 1000
            np.set_printoptions(threshold=default_numpy_truncate)
        else:
            np.set_printoptions(threshold=sys.maxsize)

        self._content.update(f"{self._value}")


class ColumnOption(Container):
    """Column which shows directory structure and selector"""

    BINDINGS = [
        Binding("left,h", "goto_parent", "Parent Directory", show=True),
        Binding("right,l", "goto_child", "Select", show=True),
    ]

    def __init__(self, fname: h5py.File, focus=False):
        super().__init__()
        self._focus = focus

        self._prev_highlighted = 0

        self._file = h5py.File(fname)
        self._cur_dir = str(self._file.name)

        self._dirs = self.get_dir_content(self._cur_dir)

    def compose(self):
        if self._dirs:
            self._header_widget = Static(f"Path: {self._cur_dir}", classes="header")
            self._selector_widget = MyOptionList(*self._dirs, id="dirs")
            self._content_widget = ColumnContent(id="content")
            yield self._header_widget
            yield self._selector_widget
            yield self._content_widget
            if self._focus:
                self._selector_widget.focus()

    def get_dir_content(self, dir) -> list[str]:
        """Return contents of current path"""
        return list(self._file[dir].keys())

    def update_list(self):
        """Redraw option list with contents of current directory"""
        self._selector_widget.clear_options()
        self._dirs = self.get_dir_content(self._cur_dir)
        self._selector_widget.add_options(self._dirs)
        self._selector_widget.highlighted = self._prev_highlighted

    def action_goto_parent(self) -> None:
        """Either displays parent or hides dataset"""
        has_parent_dir = self._cur_dir != "/"
        if has_parent_dir and not self.has_class("view-dataset"):
            self._cur_dir = os.path.dirname(self._cur_dir)
            self._header_widget.update(f"Path: {self._cur_dir}")
            self.update_list()
        self.remove_class("view-dataset")

    def action_goto_child(self) -> None:
        """Either displays child or dataset"""
        highlighted = self._selector_widget.highlighted
        if highlighted is not None:
            path = os.path.join(
                self._cur_dir,
                str(self._selector_widget.get_option_at_index(highlighted).prompt),
            )
            if path in self._file:
                if isinstance(self._file[path], h5py.Group):
                    self._prev_highlighted = highlighted
                    self._cur_dir = path
                    self._header_widget.update(f"Path: {self._cur_dir}")
                    self.update_list()
                else:
                    self.add_class("view-dataset")

                    dset = self._file[path]
                    dset_name = os.path.basename(path)
                    dset_shape = dset.shape
                    dset_values = dset[...]

                    self._content_widget.update(
                        self._cur_dir, dset_name, dset_shape, dset_values
                    )


class H5TUIApp(App):
    """Simple tui application for displaying and navigating h5 files"""

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode"), ("q", "quit", "Quit")]
    CSS_PATH = "h5tui.tcss"
    TITLE = "h5tui"

    def __init__(self, h5file):
        super().__init__()

        self._h5file = h5file

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

        with Horizontal():
            yield ColumnOption(fname=self._h5file, focus=True)

    def action_toggle_dark(self) -> None:
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )


def h5tui():
    parser = argparse.ArgumentParser(description="H5TUI")
    parser.add_argument("file", type=str, action="store", help="HDF5 File")
    args = parser.parse_args()
    h5file = args.file
    if h5py.is_hdf5(h5file):
        H5TUIApp(h5file).run()


if __name__ == "__main__":
    h5tui()
