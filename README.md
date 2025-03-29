# HoloNeRF

This project focuses on applying neural radiance fields to obtain 3D spatial data for holography
applications.

---

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Built Using](#built-using)
- [Authors](#authors)

## About

This project employs a PyTorch-based orthogonal NeRF model, and utilizes the function's density output
as a data filter to gather 3D spatial data. This is achieved by training the network on a dataset of
scene view images, and then using the trained network to predict the 3D point cloud for the given input
scene.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for
development and testing purposes.

### Python

For this project, you will need [Python 3.8](https://github.com/pyenv/pyenv) or greater. To install the
Python libraries necessary, run the installs via `pip` command.

```bash
pip install -r requirements.txt
```

### NeRF Model

This project relies on a trained NeRF model. For more information on training a NeRF model for a scene,
see the documentation.

## Usage

To generate a colored depth map of the scene, run the following command.

```bash
python3 mapping.py
```

## Built Using

- [PyTorch](https://pytorch.org/) - Deep Learning
- [NumPy](https://numpy.org/) - Matrix Manipulation
- [OpenCV](https://opencv.org/) - Image Processing
- [Open3D](http://www.open3d.org/) - 3D Data Processing

## Authors

- [@jgfranco17](https://github.com/jgfranco17)
