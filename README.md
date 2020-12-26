# Isometric Shape Interpolation

Python implementation of Baek, S., Lim, J., & Lee, K. (2015). Isometric Shape Interpolation. *Computers and Graphics*, 46(C):257--263. This is a minimalistic reimplementation of the authors' C++ code in Python. The original C++ code relied upon a bunch of 3rd party dependencies, such as SuiteSparse/CHOLMOD for sparse matrix algebra, making it extremely difficult to compile and quite hard to read. The new implementation relies on simple NumPy and SciPy codes, augmented by [Open3D](http://www.open3d.org/) and [pyquaternion](http://kieranwynn.github.io/pyquaternion/) libraries.

## Not yet available:
- Meshes with boundary. Currently, it only supports closed (watertight) meshes.
- Refinement. Section 2.3 of the paper.

## Get Started
```bash
conda create -n isoblend python=3.8 ipykernel nb_conda_kernels
conda activate isoblend
```

```bash
git clone -https://github.com/stephenbaek/isoblend.git
```

```bash
pip install -r requirements.txt
```

Run `demo.ipynb`.


## Citation
```
@article{baek2015:isometric,
    author = {Seung-Yeob Baek and Jeonghun Lim and Kunwoo Lee},
    title = {Isometric Shape Interpolation},
    journal = {Computers \& Graphics},
    volume = {46},
    number = {1},
    pages = {257--263},
    month = {2},
    year = {2015},
    url = {http://www.sciencedirect.com/science/article/pii/S0097849314001137},
    doi = {10.1016/j.cag.2014.09.025},
}
```

## License
