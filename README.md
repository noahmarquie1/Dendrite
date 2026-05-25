# Dendrite ❄️

By Noah Marquie as part of the UBC Mathematics Anthony Wachs Research Group.


2D and 3D point-cloud generation for complex meshes using physics-informed particle simulations. 

## Overview

This repository is a package to create 2D point clouds to fill in arbitrary polygons using physics-informed particle simulations. Additionally, this package includes 'extrude' functionality to extend point clouds in 3D, making hollow shells.

<img src="preview.gif" width="40%" height="40%" style="max-width: 300px" alt="Solver on simple cube.">

## Installation and Usage

Use the following two commands to install Dendrite.

```
git clone https://github.com/noahmarquie1/dendrite.git
pip install -e .
```

To generate a mesh, use the following command.

```
dendrite gen --static-poly {PATH} --dynamic-poly {PATH} --anim {WRITE_PATH} --csv {WRITE_PATH} --png {WRITE_PATH}
```

For optional flags anim, csv, and png, either a directory or full path is accepted.


## Tech Stack

- Language: Python 3.12
- Key Libraries: SciPy, JAX, NumPy, Matplotlib, Shapely
- Tools: Docker (Soon), Git (GitHub)

