
<img width="300px" alt="h5tui-logo" src="https://github.com/user-attachments/assets/f3230869-38c5-414f-8fd1-44dd48c25322" />

## Demo

![demo_plots](https://github.com/user-attachments/assets/2a93b96c-8192-4867-b2de-6dbfe230e518)

## Description

`h5tui` is a terminal user interface (TUI) for viewing the contents of [HDF5](https://www.hdfgroup.org/solutions/hdf5/) files, a binary format common in scientific computing and data science. Its design is inspired by vim-style terminal file managers such as [ranger](https://github.com/ranger/ranger), [lf](https://github.com/gokcehan/lf), and [yazi](https://github.com/sxyazi/yazi)—a natural fit because HDF5 also uses a directory-like tree of groups and datasets.

| Area | What you get |
|:-----|:-------------|
| **Browse** | Multi-column tree: groups 📁, datasets 📊; `h` / `j` / `k` / `l` and arrow keys. |
| **Search** | `/` in the current group; `f` fuzzy search across the whole file. |
| **Tables** | Scrollable grid; copy cells, jump to rows. |
| **Plots** | 1D line or histogram, 2D heatmap ([plotext](https://github.com/piccolomo/plotext)); optional PDF via `matplotlib`. |
| **Metadata** | `a` opens attributes when present. |
| **Stats** | `A` summarizes the dataset (per field for compound types). |

## Installation

The package is on [PyPI](https://pypi.org/project/h5tui/) (requires **Python ≥ 3.8.1**):

```sh
pip install h5tui
```

## Usage

```sh
h5tui file.h5
```

## Documentation

### File Navigation

`h5tui` starts at the root HDF5 group and lists the current group in the columns. **Up/down** (`k` / `j` or arrows) move the highlight; **left/right** (`h` / `l`) go to the parent group or into a child. If the selection is a **dataset**, it opens in the preview. Long lists use the same motion keys. **`~`** jumps to the file root.

While a **full-screen dataset or plot** is open, **`/`**, **`f`**, **`n`**, **`p`**, and **`~`** are disabled (search and root jump are for browse mode). With a full-screen dataset, **`j` / `k`** and paging keys in the **file columns** are disabled so navigation targets the preview. **`Esc`** in browse with no dataset open clears a sticky **`/`** highlight. In the tree, groups are shown as 📁 and datasets as 📊.

| Key | Action |
|:----|:-------|
| `h`, ← | Parent group |
| `l`, → | Enter child group or open dataset |
| `Enter` | Open highlighted entry (same idea as `l` / →) |
| `j` / `k`, ↓ / ↑ | Move cursor in the list |
| `g` / `G` | Page the column list toward start / end |
| `ctrl+u`, `u` / `ctrl+d`, `d` | Half page up / down in the column list |
| `~` | Jump to root (browse only) |
| `/` | Find in **current group** |
| `f` | **Fuzzy** find across the **entire** file |
| `n` / `p` | Next / previous find match |
| `a` | Open attributes for the selected item |
| `i` | Toggle dark mode |
| `escape` | Close plot overlay, leave dataset view, or clear sticky `/` highlights (browse) |
| `q` | Close a modal; leave dataset/plot view; or quit if already in bare browse |
| `Q` | Quit application |
| `J` / `K` | With a **file column** focused: nudge the **data table** (±10 rows) or scroll **text** preview (not used in **plot** view—plots are not vertically scrolled) |

#### Searching

| Find dialog | Keys |
|:------------|:-----|
| **`/`** (group) | Type to filter; **Enter** confirm; **Esc** / **q** cancel |
| **`f`** (fuzzy path) | **j** / **k** in the match list; **Tab** query ↔ list; **Enter** confirm; **Esc** / **q** close |

![search_demo](https://github.com/user-attachments/assets/b2d4b507-6bd7-40a6-a76b-f656b46224d9)

### Viewing Datasets

When you open a dataset, the preview is expanded. Move in the **data grid** with `h` / `j` / `k` / `l`; **`H` `J` `K` `L`** jump by **10** cells. With the grid focused, **`h` / `l`** move across columns (tree navigation uses those keys only when a file column is focused).

| Key | Action |
|:----|:-------|
| `enter` | DataTable activate / select (default) |
| `h` / `j` / `k` / `l` | Move one cell |
| `H` / `J` / `K` / `L` | Jump 10 cells |
| `u`, `pageup` / `d`, `pagedown` | Page up / down |
| `g` / `G` | Jump to top / bottom of the grid |
| `:` | **Jump to row** (plain 1D/2D table only); in the dialog, **Enter** submits, **Esc** / **q** close |
| `y` | Copy the current cell to the clipboard |

Non-table preview (scrollable **text**): **`j` / `k`**, **`u`** / **`d`** or page keys, **`g` / `G`** scroll the preview.

<img width="1512" height="900" alt="dataset table preview" src="https://github.com/user-attachments/assets/b416174e-d0c1-40f4-aea2-930c6b8a8388" />

### Plotting

`h5tui` plots in the terminal with [plotext](https://github.com/piccolomo/plotext). **1D** arrays default to a scatter-style view; use **`v`** / **`V`** to switch line plot or histogram. **2D** arrays use a **heatmap**. Dimensions above 2 are not plotted.

| Key | 1D data | 2D data |
|:----|:--------|:--------|
| `v` | Line plot | Heatmap |
| `V` | Histogram | — |
| `s` | Save plot as **PDF** (`matplotlib`) | Same |

For **compound** datatypes, choose **x** and **y** in the menu when plotting.

**Save as PDF** (`s`): in the dialog, **Esc** / **q** close, **Q** quit; **Tab** / **Enter** move among fields and **Save**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7f9804b8-e86d-4fc7-9eb9-661237575fba" width="32%"/>
  <img src="https://github.com/user-attachments/assets/3097d704-0812-4856-851f-989fe92756f9" width="32%"/>
  <img src="https://github.com/user-attachments/assets/1a5a2f43-c3a6-4baa-87cb-01d684d61c77" width="32%"/>
</p>

<img width="1511" height="908" alt="compound plot axis menu" src="https://github.com/user-attachments/assets/90f6805e-165d-435a-84bd-3a16a2ec1502" />

### Attributes

Items with attributes are marked with **▼** and a count. Press **`a`** to open a popup.

| Key | Action |
|:----|:-------|
| `escape`, `q` | Close |
| `Q` | Quit application |
| `j` / `k` | Next / previous attribute **name** |
| `J` / `K` | Scroll attribute **value** |
| `u` / `d` | Page the value up / down |

![demo_attrs](https://github.com/user-attachments/assets/8d992aa4-9a59-475b-b2e9-de40294729ef)

### Aggregation

`h5tui` can summarize datasets with **`A`** while viewing a dataset. For compound types, aggregation is **per field**. In the summary modal, **`escape`** / **`q`** close, **`Q`** quits; the stats grid uses the same keys as [Viewing datasets](#viewing-datasets).

<img width="1920" height="1048" alt="aggregation summary" src="https://user-attachments/assets/9698c153-74b2-4e8e-a433-55fe6149135e" />

### Limitations

`h5tui` is for reading HDF5 files only; it does not modify data.

## Acknowledgements

This project builds on [h5py](https://github.com/h5py/h5py), [Textual](https://github.com/Textualize/textual), [plotext](https://github.com/piccolomo/plotext), [matplotlib](https://github.com/matplotlib/matplotlib), [numpy](https://github.com/numpy/numpy), and [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz).
