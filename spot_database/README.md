# Spot Pattern Database

This folder holds the binary spot-pattern image database used by the digital
camouflage renderer (`Camo/spot_database_manager.py`, `Camo/digital_camouflage.py`):

```
spot_database/
├── large_spots/    # large spot patterns (e.g. 9x9 … 13x13 cells)
└── small_spots/    # small spot patterns (e.g. 2x2 … 6x7 cells)
```

## File format

- One binary (black/white) **PNG** image per spot pattern.
- File name encodes the spot size: `spot_<width>x<height>_<index>.png`,
  e.g. `spot_12x12_01.png`. The loader parses `<width>x<height>` from the
  file name; images are binarized with a threshold of 127 at load time.

## How to obtain

The spot database is **not open-sourced** with this repository, and no
generation script is included. Please **randomly generate the spot patterns
yourself** according to your needs (any binary PNGs following the naming
convention above will work), or **contact the authors** to request the
original database.

Both sub-folders must exist and contain at least one image each before
running the camouflage pattern generation.
