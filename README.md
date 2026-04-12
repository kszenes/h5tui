
<img width="300px" alt="h5tui-logo" src="https://github.com/user-attachments/assets/f3230869-38c5-414f-8fd1-44dd48c25322" />

**h5tui** is a terminal UI for exploring [HDF5](https://www.hdfgroup.org/solutions/hdf5/) files. Navigation feels like vim-style file managers ([ranger](https://github.com/ranger/ranger), [lf](https://github.com/gokcehan/lf), [yazi](https://github.com/sxyazi/yazi))—a natural fit, because HDF5 stores data in a tree of groups and datasets.

| Topic | Details |
|:------|:--------|
| **Mode** | Read-only: browse and inspect; files are not modified. |
| **Stack** | Python · [Textual](https://github.com/Textualize/textual) · [h5py](https://github.com/h5py/h5py) · [plotext](https://github.com/piccolomo/plotext) |

## Demo

![demo_plots](https://github.com/user-attachments/assets/2a93b96c-8192-4867-b2de-6dbfe230e518)

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [Feature summary](#feature-summary)
- [Tree and search](#tree-and-search)
- [Dataset table](#dataset-table)
- [Plots and export](#plots-and-export)
- [Attributes](#attributes)
- [Aggregation](#aggregation)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)

## Installation

Published on [PyPI](https://pypi.org/project/h5tui/). Requires **Python ≥ 3.8.1**.

```sh
pip install h5tui
```

## Usage

```sh
h5tui file.h5
```

## Feature summary

| Area | What you get |
|:-----|:-------------|
| **Browse** | Multi-column tree: groups 📁, datasets 📊; vim-style `h` / `j` / `k` / `l` plus arrows. |
| **Search** | `/` search within the current group; `f` fuzzy search across the whole file. |
| **Tables** | Dataset preview as a scrollable grid; copy cells, jump to rows. |
| **Plots** | 1D line or histogram, 2D heatmap (via [plotext](https://github.com/piccolomo/plotext)); optional PDF export with `matplotlib`. |
| **Metadata** | `a` opens attributes for the selected item. |
| **Stats** | `A` summarizes the current dataset (per field for compound types). |

---

## Tree and search

At startup you see the HDF5 root. **Up/down** (`k` / `j` or arrow keys) move inside the list; **left/right** (`h` / `l`) go to parent or child group. Selecting a **dataset** opens it in the preview. If the tree list is long, scroll it the same way. **`~`** jumps to the file root.

| Key | Action |
|:----|:-------|
| `h`, ← | Parent group |
| `l`, → | Enter child group or open dataset |
| `j` / `k`, ↓ / ↑ | Move cursor in the list |
| `~` | Jump to root |
| `/` | Find in **current group** |
| `f` | **Fuzzy** find across the **entire** file |
| `n` / `p` | Next / previous find match |
| `a` | Open attributes for the selected item |
| `i` | Toggle dark mode |
| `escape` | Leave dataset preview (back to browse) |
| `q` | Close modal / contextual quit |
| `Q` | Quit application |

![search_demo](https://github.com/user-attachments/assets/3c501334-e09c-4f9c-9f3c-d418061446c5)

---

## Dataset table

When a dataset is open, the preview is expanded. Move through the **data grid** with `h` / `j` / `k` / `l` (capital **`H` `J` `K` `L`** jump by 10 cells). Scroll long content with **`u`** / **`d`** (page up/down) where supported.

| Key | Action |
|:----|:-------|
| `h` / `j` / `k` / `l` | Move one cell |
| `H` / `J` / `K` / `L` | Jump 10 cells |
| `g` / `G` | Jump to top / bottom of the grid |
| `:` | Go to a specific **row** |
| `y` | Copy the current cell to the clipboard |

<img width="1512" height="900" alt="dataset table preview" src="https://github.com/user-attachments/assets/b416174e-d0c1-40f4-aea2-930c6b8a8388" />

---

## Plots and export

Terminal plotting uses [plotext](https://github.com/piccolomo/plotext). **1D** arrays default to a scatter-style view; toggle **line plot** or **histogram** while viewing a dataset. **2D** arrays render as a **heatmap**. Dimensions above 2 are not plotted.

| Key | 1D data | 2D data |
|:----|:--------|:--------|
| `v` | Line plot | Heatmap |
| `V` | Histogram | — |
| `s` | Save plot as **PDF** (`matplotlib`) | Same |

For **compound** datatypes, pick **x** and **y** fields from the menu when plotting.

While the plot preview is focused, **`J`** / **`K`** scroll the preview when needed.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7f9804b8-e86d-4fc7-9eb9-661237575fba" width="32%"/>
  <img src="https://github.com/user-attachments/assets/3097d704-0812-4856-851f-989fe92756f9" width="32%"/>
  <img src="https://github.com/user-attachments/assets/1a5a2f43-c3a6-4baa-87cb-01d684d61c77" width="32%"/>
</p>

<img width="1511" height="908" alt="compound plot axis menu" src="https://github.com/user-attachments/assets/90f6805e-165d-435a-84bd-3a16a2ec1502" />

---

## Attributes

Items that have HDF5 attributes show **▼** and a count. Press **`a`** to open a popup; navigation matches the rest of the UI. Close with **`q`** or **`h`** (back).

![demo_attrs](https://github.com/user-attachments/assets/8d992aa4-9a59-475b-b2e9-de40294729ef)

---

## Aggregation

Press **`A`** while viewing a dataset for a short **numerical summary**. For compound types, aggregation runs **per field**.

<img width="1920" height="1048" alt="aggregation summary" src="https://github.com/user-attachments/assets/9698c153-74b2-4e8e-a433-55fe6149135e" />

---

## Limitations

**h5tui** is read-only: it does not create, delete, or edit datasets or attributes.

## Acknowledgements

Thanks to [h5py](https://github.com/h5py/h5py), [Textual](https://github.com/Textualize/textual), [plotext](https://github.com/piccolomo/plotext), [matplotlib](https://github.com/matplotlib/matplotlib), [numpy](https://github.com/numpy/numpy), and [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) for the libraries this project builds on.
