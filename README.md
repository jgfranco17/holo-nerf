<h1 align="center">HoloNeRF</h1>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> 
    This project focuses on applying neural radiance fields to obtain 3D spatial data for holography applications.
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Deployment](#deployment)
- [Usage](#usage)
- [Built Using](#built_using)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

This project employs a PyTorch-based orthogonal NeRF model, and utilizes the function’s density output as a data filter to gather 3D spatial data. This is achieved by training the network on a dataset of scene view images, and then using the trained network to predict the 3D point cloud for the given input scene.

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 


### Python 3
For this project, you will need [Python 3.8](https://github.com/pyenv/pyenv) or greater. To install the Python libraries necessary, run the installs via `pip` command.

```bash
pip install -r requirements.txt
```

### NeRF Model

This project relies on a trained NeRF model. For more information on training a NeRF model for a scene, see the documentation.

## 🚀 Usage <a name="usage"></a>

To generate a colored depth map of the scene, run the following command.

```bash
python3 mapping.py
```

## ⛏️ Built Using <a name = "built_using"></a>

- [PyTorch](https://pytorch.org/) - Deep Learning
- [NumPy](https://numpy.org/) - Matrix Manipulation
- [OpenCV](https://opencv.org/) - Image Processing
- [Open3D](http://www.open3d.org/) - 3D Data Processing

## ✍️ Authors <a name = "authors"></a>

- [@jgfranco17](https://github.com/jgfranco17)

See also the list of [contributors](https://github.com/kylelobo/The-Documentation-Compendium/contributors) who participated in this project.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Hat tip to anyone whose code was used
- Inspiration
- References
