from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, OptionList, Static
from textual.containers import VerticalScroll, Horizontal, Container
from textual.binding import Binding

import h5py
import numpy as np

import sys
import os
import argparse


def add_escape_chars(string: str):
    return string.replace("[", r"\[")


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
        self._content = Static(markup=False)
        yield self._content

    def update(self, value):
        # save value to be able to reference it in toggle truncate
        self._value = value
        self.reprint()

    def reprint(self):
        """Used to reprint if the numpy formatting is modified"""
        self._content.update(f"{self._value}")


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
        Binding("i", "toggle_dark", "Toggle dark mode"),
        Binding("q", "quit", "Quit"),
        Binding("left,h", "goto_parent", "Parent Directory", show=True, priority=True),
        Binding("right,l", "goto_child", "Select", show=True, priority=True),
        Binding("t", "truncate_print", "Truncate print", show=False),
        Binding("s", "suppress_print", "Suppress print", show=False),
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

        self.truncate_print = True
        self.suppress_print = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

        self._header_widget = Static("/", id="header")
        yield self._header_widget
        with Horizontal():
            self._column1 = Column(self._dirs, focus=True)
            yield self._column1

    def get_dir_content(self, dir) -> list[str]:
        """Return contents of current path"""
        return list(self._file[dir].keys())

    def update_content(self, path):
        self.add_class("view-dataset")

        dset = self._file[path]
        dset_name = os.path.basename(path)
        dset_shape = dset.shape
        dset_values = dset[...]

        self._column1._content_widget.update(dset_values)

        self.update_header(f"Path: {self._cur_dir}\nDataset: {dset_name} {dset_shape}")

    def update_header(self, string):
        self._header_widget.update(string)

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
            self._column1.update_list(
                self.get_dir_content(self._cur_dir), self._prev_highlighted
            )
        self.remove_class("view-dataset")
        self.update_header(f"Path: {self._cur_dir}")

    def action_goto_child(self) -> None:
        """Either displays child or dataset"""
        highlighted = self._column1._selector_widget.highlighted
        if highlighted is not None:
            path = os.path.join(
                self._cur_dir,
                str(
                    self._column1._selector_widget.get_option_at_index(
                        highlighted
                    ).prompt
                ),
            )
            if path in self._file:
                if isinstance(self._file[path], h5py.Group):
                    self._prev_highlighted = highlighted
                    self._cur_dir = path
                    self._header_widget.update(f"Path: {self._cur_dir}")
                    self._column1.update_list(self.get_dir_content(self._cur_dir), 0)
                else:
                    self.update_content(path)

    def action_truncate_print(self):
        """Change numpy printing by toggling truncation"""
        if self.has_class("view-dataset"):
            self.truncate_print = not self.truncate_print
            if self.truncate_print:
                default_numpy_truncate = 1000
                np.set_printoptions(threshold=default_numpy_truncate)
                self.notify("Print option: Truncation = True", timeout=2)
            else:
                np.set_printoptions(threshold=sys.maxsize)
                self.notify("Print option: Truncation = False", timeout=2)
            self._column1._content_widget.reprint()

    def action_suppress_print(self):
        """Change numpy printing by suppression"""
        if self.has_class("view-dataset"):
            self.suppress_print = not self.suppress_print
            if self.suppress_print:
                np.set_printoptions(suppress=True)
                self.notify("Print option: Suppress = True", timeout=2)
            else:
                np.set_printoptions(suppress=False)
                self.notify("Print option: Suppress = False", timeout=1)
            self._column1._content_widget.reprint()


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
