
<img width="300px" alt="h5tui-logo" src="https://github.com/user-attachments/assets/f3230869-38c5-414f-8fd1-44dd48c25322" />

## Demo

![demo_plots](https://github.com/user-attachments/assets/2a93b96c-8192-4867-b2de-6dbfe230e518)

## Description

`h5tui` is a terminal user interface (TUI) for viewing the contents of [HDF5](https://www.hdfgroup.org/solutions/hdf5/) files, a binary format common in scientific computing and data science. Its design is inspired by vim-style terminal file managers such as [ranger](https://github.com/ranger/ranger), [lf](https://github.com/gokcehan/lf), and [yazi](https://github.com/sxyazi/yazi)—a natural fit because HDF5 also uses a directory-like tree of groups and datasets.

- **Browse:** Three-column tree view of groups 📁 and datasets 📊 with Vim keybindings. Fuzzy search for quick navigation.
- **Plots:** 1D line plot or histogram, 2D heatmap with [plotext](https://github.com/piccolomo/plotext); PDF export via [matplotlib](https://github.com/matplotlib/matplotlib).

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

`h5tui` starts at the root HDF5 group. Groups are 📁, datasets 📊.

| Key | Action |
|:----|:-------|
| `Enter` | Open the highlighted entry (same as `l` / →) |
| `~` | Jump to root |
| `a` | Open **attributes** for the selection |
| `i` | Toggle **dark mode** |

#### Searching

| Dialog | Extra detail |
|:-------|:-------------|
| `/` (group) | Type to filter; `Enter` confirms (see shared `Esc` / `q`). |
| `n` / `p` | Next / previous find match |
| `f` (fuzzy path) | `Tab` switches between query and match list. |

![search_demo](https://github.com/user-attachments/assets/b2d4b507-6bd7-40a6-a76b-f656b46224d9)

### Viewing Datasets

<img width="1512" height="900" alt="dataset table preview" src="https://github.com/user-attachments/assets/b416174e-d0c1-40f4-aea2-930c6b8a8388" />

| Key | Action |
|:----|:-------|
| `:` | Jump to row (plain 1D/2D table only); in the dialog, `Enter` submits (see shared dismiss keys to cancel). |
| `y` | Copy the focused cell |

### Plotting

#### Inline

`h5tui` plots in the terminal with [plotext](https://github.com/piccolomo/plotext). **1D** defaults to a scatter-style view. **2D** uses a **heatmap**. Dimensions above **2** are not plotted.

| Key | 1D | 2D |
|:----|:---|:---|
| `v` | Line plot | Heatmap |
| `V` | Histogram | — |

For **compound** datatypes, choose **x** and **y** in the menu.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7f9804b8-e86d-4fc7-9eb9-661237575fba" width="32%"/>
  <img src="https://github.com/user-attachments/assets/3097d704-0812-4856-851f-989fe92756f9" width="32%"/>
  <img src="https://github.com/user-attachments/assets/1a5a2f43-c3a6-4baa-87cb-01d684d61c77" width="32%"/>
</p>

#### Export as PDF

Press `s` to open dialog to **Save Plot As PDF**  via [matplotlib](https://github.com/matplotlib/matplotlib).

<img width="1511" height="908" alt="compound plot axis menu" src="https://github.com/user-attachments/assets/90f6805e-165d-435a-84bd-3a16a2ec1502" />

### Attributes

Press `a` from the tree. Items with metadata show **▼** and a count.

![demo_attrs](https://github.com/user-attachments/assets/8d992aa4-9a59-475b-b2e9-de40294729ef)

### Aggregation

Press `A` while viewing a dataset for a **numerical summary** (per field for compound types).

<img width="1920" height="1048" alt="aggregation summary" src="https://github.com/user-attachments/assets/9698c153-74b2-4e8e-a433-55fe6149135e" />

### Shared navigation

Most screens reuse the same motion and dismissal keys; meaning depends on **focus** (tree vs table vs modal).

| Pattern         | Keys                                  |
|:--------------- |:--------------------------------------|
| **Move**        | `h` `j` `k` `l`, arrows               |
| **Jump 10 grid cells** | `H` `J` `K` `L`                       |
| **Start/end**   | `g` `G`                               |
| **Page**        | `u` `d`, `PageUp` `PageDown`          |
| **Dismiss/quit**| `Esc`, `q`, `Q`                       |

### Limitations

`h5tui` is meant for reading HDF5 files only; it does not provide functionality for modifying them.

## Acknowledgements

This project builds on [h5py](https://github.com/h5py/h5py), [Textual](https://github.com/Textualize/textual), [plotext](https://github.com/piccolomo/plotext), [matplotlib](https://github.com/matplotlib/matplotlib), [numpy](https://github.com/numpy/numpy), and [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz).
