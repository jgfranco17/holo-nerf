# Project Background

A general overview of the research project.

## Abstract

In this paper, we propose a novel approach for obtaining 3D point cloud data from a set of images using neural radiance fields (NeRF). Our method employs a PyTorch-based orthogonal NeRF model, and utilizes the function’s density output as a data filter to gather 3D spatial data. This is achieved by training the network on a dataset of scene view images, and then using the trained network to predict the 3D point cloud for the given input scene. The generated data is highly realistic and has a high degree of detail, making it suited for use in creating holograms. The proposed method is evaluated on a range of images, and the results show that it can effectively generate 3D data that is suitable for use in holography applications. Overall, this work presents a promising approach for generating 3D spatial data for holography using NeRF.

## Core Concepts

### Light Fields

Light fields are vector functions that convert the geometry of light rays into plenoptic properties. The light field is the sum of all light rays in 3D space, travelling through all points and directions. The five-dimensional plenoptic function defines the space of all conceivable light rays, and the magnitude of each ray is provided by its brightness.

### Neural Radiance Fields

A neural radiance field (NeRF) is a fully-connected neural network that can generate novel views of complex 3D scenes, based on a partial set of 2D images. NeRFs are based on the concept of a light fields, which describes the spatial and angular distribution of light in a scene. Specifically, an NeRF is a function that maps a 3D coordinates and the 2 viewing directions of a scene point `(x, y, z, θ, ϕ)` as input to a 4D output of point color `(r, g, b, σ)`, where the first three are the RGB color and the fourth dimension is the color density. NeRFs use machine learning techniques, particularly deep neural networks, to learn the relationship between 3D scenes and the corresponding light fields.

NeRFs have the ability to generate high-quality images of 3D scenes with a wide range of lighting conditions and viewpoints. They can also handle complex geometric structures and materials, such as transparent and reflective objects. NeRFs have been used in a variety of applications, including virtual and augmented reality, computer-generated imagery, and 3D printing. One of the main advantages of NeRFs is their ability to generate images on the fly, allowing for interactive rendering of 3D scenes. However, NeRFs can be computationally expensive to train and evaluate, requiring significant amounts of data and computing resources as each NeRF model is unique to it's scene.