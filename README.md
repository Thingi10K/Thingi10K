## Thingi10K Dataset

![Thingi10K Poster](https://user-images.githubusercontent.com/3606672/65047743-fa269180-d930-11e9-8013-134764b150c1.png)

Thingi10K is a large scale 3D dataset created to study the variety, complexity and quality of
real-world 3D printing models. We analyze every mesh of all things featured on
[Thingiverse.com](https://www.thingiverse.com/)
between Sept. 16, 2009 and Nov. 15, 2015. On this site, we hope to share our findings with you.

In a nutshell, Thingi10K contains...

* 10,000 models
* 4,892 tags
* 2,011 things
* 1,083 designers
* 72 categories
* 10 open source licenses
* 7+ years span
* 99.6% .stl files
* 50% non-solid
* 45% with self-intersections
* 31% with coplanar self-intersections
* 26% with multiple components
* 22% non-manifold
* 16% with degenerate faces
* 14% non-PWN
* 11% topologically open
* 10% non-oriented

Thingi10K is created by [Qingnan Zhou](https://research.adobe.com/person/qingnan-zhou/) and [Alec
Jacobson](http://www.cs.toronto.edu/~jacobson/).

## Raw dataset

You can download the raw dataset from one of the following mirrors:
[NYU Box](https://nyu.app.box.com/s/n1znd0u5blvsua2txsn6i942ltthhd4u),
[Hugging Face](https://huggingface.co/datasets/Thingi10K/Thingi10K/blob/main/Thingi10K.tar.gz),
[Google Drive](https://drive.google.com/file/d/1RlDvNiFLDRztN0zWqQxmeraRG-XXFHUT/view).

One can also obtain the dataset via the `thingi10k` Python package. It contains both geometric and
contextual data extracted from the raw dataset, and provides a convenient API to access and filter
the dataset.

## Usage

In addition to the raw dataset, we provide a Python package `thingi10k` to facilitate easy access to
the dataset. The package provides functions to download, filter, and load the dataset.

### Installation

```sh
pip install thingi10k
```

### Simple usage

```py
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "thingi10k",
# ]
# ///

import thingi10k

thingi10k.init() # Download the dataset and update cache

# Loop through all entries in the dataset
for entry in thingi10k.dataset():
    file_id = entry['file_id']
    author = entry['author']
    license = entry['licence']
    vertices, facets = thingi10k.load_file(entry['file_path'])
    # Do something with the vertices and facets

help(thingi10k) # for more information
```

### Filtering the dataset

The `thingi10k.dataset()` function provides a convenient way to filter the dataset based on various
geometric and contextual criteria. The function returns an iterator over the filtered entries. The
following are some examples of filtering the dataset:

The example below demonstrates how to iterate over models in the Thingi10K dataset that are
closed and have at most 100 vertices.

```py
for entry in thingi10k.dataset(num_vertices=(None, 100), closed=True):
    vertices, facets = thingi10k.load_file(entry['file_path'])
```

The following example shows how to filter and iterate over models that are licensed under Creative
Commons.

```py
for entry in thingi10k.dataset(license='creative commons'):
    vertices, facets = thingi10k.load_file(entry['file_path'])
```

This example illustrates how to iterate over models that are solid, consist of a single component,
and have no self-intersections.

```py
for entry in thingi10k.dataset(num_components=1, self_intersecting=False, solid=True):
    vertices, facets = thingi10k.load_file(entry['file_path'])
```

Please see `help(thingi10k.dataset)` for all available filtering options.

### Dataset variants

Thingi10K provides two variants of the dataset: `npz` and `raw`.

* `npz` variant contains the geometry (vertex and facet arrays) in NumPy arrays. It is faster to
download and no mesh parsing is necessary.
* `raw` variant contains the raw mesh files (STL, OBJ, etc.) in their original format. It is slower
to download and requires parsing to extract geometric data.

By default, `thingi10k.init()` will download the `npz` variant. To download the `raw` variant:

```py
thingi10k.init(variant='raw')
```

### Caching the dataset

By default, `thingi10k.init()` will cache the dataset in a local directory.
Any subsequent calls to `thingi10k.init()` will use the cached dataset and incur no additional
download cost.
The cache directory can be explicitly specified by user:

```py
thingi10k.init(cache_dir="path/to/.thingi10k")
```

To force a re-download of the dataset:

```py
thingi10k.init(force_redownload=True)
```


## License

The source code for organizing and filtering the Thingi10K dataset is licensed under the Apache License,
Version 2.0. Each "thing" in the dataset is licensed under different licenses. Please refer to the
`license` field associated with each entry in the dataset.

## Errata

The following models are known to be "corrupt." However, we decide to still include them in our dataset in order to faithfully reflect mesh qualities on Thingiverse.

* Model 49911 is truncated (ASCII STL).
* Model 74463 is empty.
* Model 286163 is empty.
* Model 81313 contains NURBS curves and surfaces instead of polygonal faces, which may not be supported by many OBJ parsers.
* Model 77942 is corrupt (binary STL).

## Acknowledgements

This project is funded in part by NSF grants CMMI-11-29917, IIS-14-09286, and IIS-17257.

We thank Marcel Campen, Chelsea Tymms, and Julian Panetta for early feedback and proofreading.
We also thank Neil Dickson for pointing out corrupt models, and Nick Sharp for pointing out bugs in download script.
Lastly, we thank Silvia Sellán and Yun-Chun Chen for discussion and suggestion on hosting the dataset.

## Cite us

```bibtex
@article{Thingi10K,
  title={Thingi10K: A Dataset of 10,000 3D-Printing Models},
  author={Zhou, Qingnan and Jacobson, Alec},
  journal={arXiv preprint arXiv:1605.04797},
  year={2016}
}
```
