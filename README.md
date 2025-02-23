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

## Plotting

`h5tui` provides convenient terminal plotting facilities using the [plotext](https://github.com/piccolomo/plotext) library.
1D arrays are displayed as scatter plots, and 2D arrays are shown as heatmaps. Higher dimensional tensors are not currently supported.
The plotting can be toggled through the `p` keybinding while viewing a dataset.

## Aggregation

`h5tui` also has limited data aggregation facilities for summarizing datasets.
This can be activated through the `a` keybinding while viewing a dataset.
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

## Limitations

- There is no support for displaying HDF5 attributes (mostly because the HDF5 files that I work with don't rely on them). However, if there is demand, this functionality can be added.
- There is no editing functionality, the contents of the HDF5 file cannot be modified through `h5tui`.
- I have only tested  dataset viewing and plotting for primitive types (strings, ints, floats) and arrays. Please let me know if you encounter any issues.
