
<img width="300px" alt="h5tui-logo" src="https://github.com/user-attachments/assets/f3230869-38c5-414f-8fd1-44dd48c25322" />

## Description

`h5tui` is a terminal user interface (TUI) application for viewing the contents of [HDF5](https://www.hdfgroup.org/solutions/hdf5/) files, a binary file format prevalent in scientific computing and data science, straight in the terminal.
Its design is inspired by vim-motion-enabled terminal file managers, such as [ranger](https://github.com/ranger/ranger), [lf](https://github.com/gokcehan/lf) and [yazi](https://github.com/sxyazi/yazi).
This choice is natural since the HDF5 file format also adopts a directory structure for storing the data.

This project wouldn't have been possible without [h5py](https://github.com/h5py/h5py) for reading the HDF5 files, [textual](https://github.com/Textualize/textual) for building the UI, and [plotext](https://github.com/piccolomo/plotext) for plotting data straight in the terminal. 

## Demo

https://github.com/user-attachments/assets/356225e7-e2ab-457a-8e47-97c19efb5aaa

## Installation

The package is hosted on [PyPI](https://pypi.org/project/h5tui/) and can be installed using `pip`:

```sh
pip install h5tui
```

## Usage

Simply launch the application with an HDF5 file as an argument:

```sh
h5tui file.h5
```

## File Navigation

`h5tui` starts at the root of the file and displays the contents of the root HDF5 group.
The directory structure can be navigated using the arrow or standard vim motion keys, with the `up`/`down` (`j`/`k`) moving the cursor inside the current group, and `left`/`right` (`h`/`l`) for going to the parent or child HDF5 group.
If the selected element is not an HDF5 group but an HDF5 dataset, then the dataset is displayed.
If the entire dataset does not fit on the screen, it can be scrolled using the `up`/`down` `j`/`k` keybindings.

## Attributes

Items with associated attributes are marked with the symbol ▼ followed by the number of attributes. They can be viewed using the `a` character, which will open a popup with the attributes for that item. The navigation in this menu is analogous to the general menu. The menu can be exited using the `q` or `left,h` keys.

https://github.com/user-attachments/assets/757768e2-77fa-4708-ba17-a334e299fdfd


## Plotting

`h5tui` provides convenient terminal plotting facilities using the [plotext](https://github.com/piccolomo/plotext) library.
1D arrays are displayed as scatter plots, and 2D arrays are shown as heatmaps. Higher dimensional tensors are not currently supported.
The plotting can be toggled through the `p` keybinding while viewing a dataset.

## Aggregation

`h5tui` also has limited data aggregation facilities for summarizing datasets.
This can be activated through the `A` keybinding while viewing a dataset.
Currently, this option will compute the min, max, and mean of the dataset but further statistics may be added in the future.

## Dataset Format Options

The formatting of the dataset may be controlled using a couple of keybindings.
Since HDF5 files often contain large datasets which, by default, will truncate the output if the number of elements exceeds 1000 (that is the `numpy` default).
This behavior can be `t`oggled using the `t` keybinding to display the entire dataset.
Note that, currently, this operation is blocking, and therefore huge datasets might take some time to load.
In addition, the `s` key toggles the scientific notation on and off (corresponding to the `suppress` option in `numpy`s printing configuration).

Formatting keybindings:
- `t`: toggle output truncation
- `s`: toggle scientific notation

## Special Treatment of Dataframes

Dataframes, which `h5py` stores in the form of an `np.recarray` are pretty-printed using the `textual` `DataTable` widget. The table can be navigated vertically using the `k`/`j` keys and horizontally using the capital keys `H`,`L`, to avoid clashing with the `h` key which would take you back to the directory structure.

![image](https://github.com/user-attachments/assets/eaf07aad-ffba-483e-8740-d9ac85fa6eab)

## Limitations

- There is no editing functionality, the contents of the HDF5 file cannot be modified through `h5tui`.
- I have only tested  dataset viewing and plotting for primitive types (strings, ints, floats) and arrays. Please let me know if you encounter any issues.
