<img width="300px" alt="h5tui-logo" src="https://github.com/user-attachments/assets/f3230869-38c5-414f-8fd1-44dd48c25322" />

## Demo

![demo_plots](https://github.com/user-attachments/assets/2a93b96c-8192-4867-b2de-6dbfe230e518)

## Description

`h5tui` is a vim-style TUI for browsing [HDF5](https://www.hdfgroup.org/solutions/hdf5/) files, inspired by terminal managers like [ranger](https://github.com/ranger/ranger), [lf](https://github.com/gokcehan/lf), and [yazi](https://github.com/sxyazi/yazi).

- **Browse:** **Three-column tree view** of HDF5 files with **Vim-style keybindings**. Includes **fuzzy search** for quick navigation.
- **Plots:** **Inline** 1D **line plot** or **histogram**, 2D **heatmap** with [plotext](https://github.com/piccolomo/plotext) or **PDF export** via [matplotlib](https://github.com/matplotlib/matplotlib).

## Installation

The package is on [PyPI](https://pypi.org/project/h5tui/):

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

| Key                   | Action                            |
| :-------------------- | :-------------------------------- |
| `h` / `j` / `k` / `l` | **Move** left / down / up / right |
| `g` / `G`             | **Jump** to top / bottom          |
| `d` / `u`             | **Page** down / up                |
| `~`                   | Return to **root**                |

#### Searching

| Key       | Action                        |
| :-------- | :---------------------------- |
| `/`       | **Find** in group             |
| `n` / `p` | **Next** / **previous** match |
| `f`       | **Fuzzy** search entire file  |

![search_demo](https://github.com/user-attachments/assets/b2d4b507-6bd7-40a6-a76b-f656b46224d9)

### Viewing Datasets

Datasets are opened in a viewer that supports the same type of VIM keybindings for navigating the cells.

<img width="1512" height="900" alt="dataset table preview" src="https://github.com/user-attachments/assets/b416174e-d0c1-40f4-aea2-930c6b8a8388" />

Additional keybindings:

| Key | Action                             |
| :-- | :--------------------------------- |
| `:` | **Jump to row**                    |
| `y` | **Copy** focused cell to clipboard |

### Plotting

#### Inline

`h5tui` plots in the terminal with [plotext](https://github.com/piccolomo/plotext).

<p align="center">
  <img src="https://github.com/user-attachments/assets/7f9804b8-e86d-4fc7-9eb9-661237575fba" width="32%"/>
  <img src="https://github.com/user-attachments/assets/3097d704-0812-4856-851f-989fe92756f9" width="32%"/>
  <img src="https://github.com/user-attachments/assets/1a5a2f43-c3a6-4baa-87cb-01d684d61c77" width="32%"/>
</p>

| Key | 1D        | 2D      | >2D |
| :-- | :-------- | :------ | :-- |
| `v` | Line plot | Heatmap | —   |
| `V` | Histogram | —       | —   |

For **compound** datatypes, any field can be chosen as the x and y axis.

#### Export as PDF

Press `s` to open dialog to **Save Plot As PDF** via [matplotlib](https://github.com/matplotlib/matplotlib).

<img width="1511" height="908" alt="compound plot axis menu" src="https://github.com/user-attachments/assets/90f6805e-165d-435a-84bd-3a16a2ec1502" />

### Attributes

Items with attributes display **▼** and a (count).
Press `a` to view them.

![demo_attrs](https://github.com/user-attachments/assets/8d992aa4-9a59-475b-b2e9-de40294729ef)

### Aggregation

Press `A` while viewing a dataset for a **numerical summary** including (e.g., mean, variance, max).
This is performed per field for compound types.

<img width="1920" height="1048" alt="aggregation summary" src="https://github.com/user-attachments/assets/9698c153-74b2-4e8e-a433-55fe6149135e" />

### Shared navigation

Most screens reuse the same motion and dismissal keys; meaning depends on **focus** (tree vs table vs modal).

| Pattern                | Keys                         |
| :--------------------- | :--------------------------- |
| **Move**               | `h` `j` `k` `l`, arrows      |
| **Jump 10 grid cells** | `H` `J` `K` `L`              |
| **Start/end**          | `g` `G`                      |
| **Page**               | `u` `d`, `PageUp` `PageDown` |
| **Dismiss/quit**       | `Esc`, `q`, `Q`              |

### Limitations

`h5tui` is meant for reading HDF5 files only; it does not provide functionality for modifying them.

## Acknowledgements

This project builds on [h5py](https://github.com/h5py/h5py), [Textual](https://github.com/Textualize/textual), [plotext](https://github.com/piccolomo/plotext), [matplotlib](https://github.com/matplotlib/matplotlib), [numpy](https://github.com/numpy/numpy), and [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz).
