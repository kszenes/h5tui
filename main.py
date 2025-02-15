from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Footer, Header, OptionList, Static
from textual.containers import VerticalScroll, Horizontal
from textual.binding import Binding
from textual.reactive import reactive
import argparse

import h5py
import os


class MyOptionList(OptionList):
    BINDINGS = [
        Binding("down,j", "cursor_down", "Down", show=True),
        Binding("up,k", "cursor_up", "Up", show=True),
        Binding("G", "page_down", "Page down", show=False),
        Binding("g", "page_up", "Page up", show=False),
    ]


class ColumnContent(VerticalScroll):
    """Column which displays a dataset"""
    def compose(self):
        self._content = Static()
        yield self._content

    def update(self, value):
        self._content.update(f"{value}")


class ColumnOption(Widget):
    """Column which shows directory structure and selector"""
    BINDINGS = [
        ("left,h", "goto_parent", "Parent Directory"),
        ("right,l", "goto_child", "Select"),
    ]

    def __init__(self, fname: h5py.File, focus=False):
        super().__init__()
        self._focus = focus
        self._dir_hidden = False

        self._prev_highlighted = 0

        self._file = h5py.File(fname)
        self._cur_dir = str(self._file.name)

        self._dirs = self.get_dir_content(self._cur_dir)

    def compose(self):
        if self._dirs:
            self._path_widget = Static(f"Path: {self._cur_dir}")
            self._list_widget = MyOptionList(*self._dirs, id="dirs")
            self._content_widget = ColumnContent(id="content")
            yield self._path_widget
            yield self._list_widget
            yield self._content_widget
            if self._focus:
                self._list_widget.focus()

    def get_dir_content(self, dir) -> list[str]:
        """Return contents of current path"""
        return list(self._file[dir].keys())

    def update_list(self):
        """Redraw option list with contents of current directory"""
        self._list_widget.clear_options()
        self._dirs = self.get_dir_content(self._cur_dir)
        self._list_widget.add_options(self._dirs)
        self._list_widget.highlighted = self._prev_highlighted

    def action_goto_parent(self) -> None:
        """Either displays parent or hides dataset"""
        has_parent_dir = self._cur_dir != '/'
        if has_parent_dir and not self._dir_hidden:
            self._cur_dir = os.path.dirname(self._cur_dir)
            self._path_widget.update(f"Path: {self._cur_dir}")
            self.update_list()
        self.remove_class("view-dataset")
        self._dir_hidden = False

    def action_goto_child(self) -> None:
        """Either displays child or dataset"""
        highlighted = self._list_widget.highlighted
        if highlighted is not None:
            path = os.path.join(
                self._cur_dir,
                str(self._list_widget.get_option_at_index(highlighted).prompt),
            )
            if path in self._file:
                if isinstance(self._file[path], h5py.Group):
                    self._prev_highlighted = highlighted
                    self._cur_dir = path
                    self._path_widget.update(f"Path: {self._cur_dir}")
                    self.update_list()
                else:
                    self.add_class("view-dataset")
                    self._dir_hidden = True

                    self._content_widget.update(self._file[path][...])


class H5TUI(App):
    """Simple tui application for displaying and navigating h5 files"""

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode"), ("q", "quit", "Quit")]
    CSS_PATH = "h5tui.tcss"

    def __init__(self, h5file):
        super().__init__()

        self._h5file = h5file

    def compose(self) -> ComposeResult:
        yield Header(name="H5TUI")
        yield Footer()

        with Horizontal():
            yield ColumnOption(fname=self._h5file, focus=True)

    def action_toggle_dark(self) -> None:
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H5TUI")
    parser.add_argument("file", type=str, action="store", help="HDF5 File")
    args = parser.parse_args()
    h5file = args.file
    if h5py.is_hdf5(h5file):
        H5TUI(h5file).run()
