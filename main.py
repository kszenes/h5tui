from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, DirectoryTree, OptionList
from textual.containers import VerticalScroll, Horizontal

import h5py
import os


class Column(VerticalScroll):
    def __init__(self, dir=None, focus=False):
        super().__init__()

        self._focus = focus

        if dir is not None:
            self._dirs = [str(i) for i in os.listdir(dir)]
        else:
            self._dirs = None

    def compose(self):
        # yield DirectoryTree("./")
        if self._dirs:
            self._list = OptionList(*self._dirs)
            yield self._list
            if self._focus:
                self._list.focus()


class H5TUI(App):
    """Simple tui application for displaying and navigating h5 files"""

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def compose(self) -> ComposeResult:
        yield Header(name="H5TUI")
        yield Footer()

        with Horizontal():
            yield Column(dir="..")
            yield Column(dir=".", focus=True)
            yield Column()



    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode"""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )


if __name__ == "__main__":
    H5TUI().run()
