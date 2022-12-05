The modules in this directory contain implementations of different monocular 3D face reconstruction algorithms. To work well with this package, each algorithm is implemented as a class with at least the following methods:

* `__call__`, which receives a single argument (a 2D numpy array representing an RGB image) and performs the reconstruction; should return a dictionary with at least the key `"v"` with the reconstructed vertices as values
