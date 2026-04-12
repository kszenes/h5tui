
<img width="300px" alt="h5tui-logo" src="https://github.com/user-attachments/assets/f3230869-38c5-414f-8fd1-44dd48c25322" />

## Demo

![demo_plots](https://github.com/user-attachments/assets/2a93b96c-8192-4867-b2de-6dbfe230e518)

## Description

`h5tui` is a terminal user interface (TUI) application for viewing the contents of [HDF5](https://www.hdfgroup.org/solutions/hdf5/) files, a binary file format prevalent in scientific computing and data science, straight in the terminal.
Its design is inspired by vim-motion-enabled terminal file managers, such as [ranger](https://github.com/ranger/ranger), [lf](https://github.com/gokcehan/lf) and [yazi](https://github.com/sxyazi/yazi).
This choice is natural since the HDF5 file format also adopts a directory structure for storing the data.


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

## Documentation

### File Navigation

`h5tui` starts at the root of the file and displays the contents of the root HDF5 group.
The directory structure can be navigated using the arrow or standard vim motion keys, with the `up`/`down` (`j`/`k`) moving the cursor inside the current group, and `left`/`right` (`h`/`l`) for going to the parent or child HDF5 group.
If the selected element is not an HDF5 group but an HDF5 dataset, then the dataset is displayed.
If the entire dataset does not fit on the screen, it can be scrolled using the `up`/`down` `j`/`k` keybindings.
Inside the file tree, Groups are denoted by 📁 while Datasets are denoted by 📊.
To quickly return to the root of the file, press `~`.

#### Searching

Use `/` to search for an entry within a group and `f` fuzzy searches through the entire file

![search_demo](https://github.com/user-attachments/assets/3c501334-e09c-4ab6-9f3c-d418061446c5)

### Viewing Datasets

When entering a dataset, the preview is maximized and the keys `h,j,k,l` can be used to navigate within the dataset (the capital keys `H,J,K,L` jump by 10 cells).
The keys `G` and `g` can be used to jump to the beginning and end of the file, respectively, while the `:` key can be used to navigate to a specific row.
The content of the currently selected cell can be copied to clipboard using `y`.

<img width="1512" height="900" alt="image" src="https://github.com/user-attachments/assets/b416174e-d0c1-40f4-aea2-930c6b8a8388" />

### Plotting

`h5tui` provides convenient terminal plotting facilities using the [plotext](https://github.com/piccolomo/plotext) library.
1D arrays are displayed as scatter plots, and 2D arrays are shown as heatmaps. Higher dimensional tensors are not currently supported.
The plotting can be toggled through the `v/V` (line plot for 1d ) keybinding while selecting or viewing a dataset.
For 1D data `v` plots a lineplot while `V` a histogram, for 2D data, `v` generates a heatmap.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7f9804b8-e86d-4fc7-9eb9-661237575fba" width="32%"/>
  <img src="https://github.com/user-attachments/assets/3097d704-0812-4856-851f-989fe92756f9" width="32%"/>
  <img src="https://github.com/user-attachments/assets/1a5a2f43-c3a6-4baa-87cb-01d684d61c77" width="32%"/>
</p>

Or save as pdf using `matplotlib` with the `s` key.
For compound datatypes, the x and y axis can be chosen from the menu.

<img width="1511" height="908" alt="image" src="https://github.com/user-attachments/assets/90f6805e-165d-435a-84bd-3a16a2ec1502" />


### Attributes

Items with associated attributes are marked with the symbol ▼ followed by the number of attributes. They can be viewed using the `a` character, which will open a popup with the attributes for that item. The navigation in this menu is analogous to the general menu. The menu can be exited using the `q` or `left,h` keys.

![demo_attrs](https://github.com/user-attachments/assets/8d992aa4-9a59-475b-b2e9-de40294729ef)


### Aggregation

`h5tui` also has limited data aggregation facilities for summarizing datasets.
This can be activated through the `A` keybinding while viewing a dataset.
For compound datatypes, this aggregation is performed per field.

<img width="1920" height="1048" alt="image" src="https://github.com/user-attachments/assets/9698c153-74b2-4e8e-a433-55fe6149135e" />


### Limitations

`h5tui` is for reading HDF5 files and does not provide support for modifying the data.

## Acknowlegements

This project wouldn't have been possible without [h5py](https://github.com/h5py/h5py) for reading the HDF5 files, [textual](https://github.com/Textualize/textual) for building the UI, [plotext](https://github.com/piccolomo/plotext) for plotting data straight in the terminal and [matplotlib](https://github.com/matplotlib/matplotlib) for generating PDF plots, [numpy](https://github.com/numpy/numpy) for the numerics and [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) for the fuzzy searching.
