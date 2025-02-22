## Description

`h5tui` is a terminal user interface (TUI) application that facilitates the viewing of the contents of HDF5 files.
Its design is inspired by vim-motion enabled terminal file managers, such as [ranger](https://github.com/ranger/ranger), [lf](https://github.com/gokcehan/lf) and [yazi](https://github.com/sxyazi/yazi).
This choice is quite natural since the HDF5 file format also adopts a directory structure for storing the data.

## Demo

https://github.com/user-attachments/assets/11085d02-2857-4d9f-aac1-b403630b8ead

## Installation

The package is hosted and PyPI and can be installed using `pip`:

```sh
pip install h5tui
```

## Usage

Simply launch the application with an HDF5 file as an argument:

```sh
h5tui file.h5
```

## File Navigation

`h5tui` starts at the root of the file and displays the contents of the directory.
The file can be navigated using the arrow or standard vim motion keys, with the `up`/`down` (`j`/`k`) moving the cursor inside the list, and `left`/`right` (`h`/`l`) for going to the parent or child HDF5 group.
If the selected element is not an HDF5 group but an HDF5 dataset, then the contents of the dataset are displayed.
If the data does not fit on one screen, it can be scrolled using the `up`/`down` `j`/`k` keybindings.

## Plotting

`h5tui` provides simple plotting facilities straight in the terminal using the [Plotext](https://github.com/piccolomo/plotext) library.
1D arrays are displayed as scatter plots, and 2d arrays are shown as heatmaps. Higher dimensional tensors are not currently supported.
The plotting can be toggled through the `p` keybinding while viewing a dataset.

## Aggregation

`h5tui` also has the functionality for performing limited data aggregation for summarizing the data.
This can be activated through the `a` keybinding while viewing a dataset.
Currently, this option will compute the min, max, and mean of the dataset but further statistics may be added in the future.

## Dataset Format Options

The formatting of the dataset may be controlled using a couple of keybindings.
Since HDF5 files can often contain large datasets, by default, if the number of elements exceeds 1000, the output will be truncated.
This behavior can be `t`oggled using the `t` keybinding.
Note that, as of yet, this operation is blocking, and therefore huge datasets might take some time to load
In addition, the `s` key toggles the scientific notation on and off.

Formatting keybindings:
- `t`: toggle output truncation
- `s`: toggle scientific notation
