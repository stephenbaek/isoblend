# Isometric Shape Interpolation

Python implementation of Baek, S., Lim, J., & Lee, K. (2015). Isometric Shape Interpolation. *Computers and Graphics*, 46(C):257--263. This is a minimalistic reimplementation of the authors' C++ code in Python. The original C++ code relied upon a bunch of 3rd party dependencies, such as SuiteSparse/CHOLMOD for sparse matrix algebra, making it extremely difficult to compile and quite hard to read. The new implementation relies on simple NumPy and SciPy codes, augmented by [Open3D](http://www.open3d.org/) and [pyquaternion](http://kieranwynn.github.io/pyquaternion/) libraries.

## Not yet available:
- Meshes with boundary. Currently, it only supports closed (watertight) meshes.
- Speed. It is far slower than the original C++ code.

The above issues will be addressed in the near future.

## Get Started
Clone this repository by running the following comand.
```bash
git clone -https://github.com/stephenbaek/isoblend.git
cd isoblend
```
Create a virtual environment using conda.
Alternatively, you could use `virtualenv`.
```bash
conda create -n isoblend python=3.8 ipykernel nb_conda_kernels
conda activate isoblend
```

Install dependencies.
```bash
pip install -r requirements.txt
```

You are now ready to run the code. For the usage, see `demo.ipynb`.


## Citation
If you use this code for your work, please cite the article below.
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

```
Copyright 2020 Stephen Baek

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```