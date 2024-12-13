## Thingi10K Dataset

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

## Installation

```sh
pip install thingi10k
```

## Usage

```py
import thingi10k

thingi10k.init() # Initial download of the dataset


# Iterate over the entire dataset
for entry in thingi10k.dataset():
    file_id = entry['file_id']

    # Check contextual data
    author = entry['author']
    license = entry['license']

    # Load actual geometry
    vertices, facets = thingi10k.load_file(entry['file_path'])
```

### Caching the dataset

By default, `thingi10k.init()` will cache the dataset in a local directory.
Any subsequent calls to `thingi10k.init()` will use the cached dataset and incur no additional
download cost.
The cache directory can be explicitly specified by user:

```py
thigni10k.init(cache_dir="path/to/.thingi10k")
```

To force a re-download of the dataset:

```py
thingi10k.init(force_redownload=True)
```

### Filtering the dataset

Thingi10K supports advanced filtering of the dataset based on geometric and contextual properties.
See `help(thingi10k.dataset)` for a full list of supported filters. Here are some examples:

```py
# Filter by closed models
closed_models = thingi10k.dataset(closed=True)

# Filter by models without self-intersections
no_self_intersections = thingi10k.dataset(self_intersecting=False)

# Filter by the number of vertices (<100)
small_models = thingi10k.dataset(num_vertices=(None, 100))

# Models with a single component and is manifold
single_manifold_models = thingi10k.dataset(num_components=1, manifold=True)

# Models licensed under the Creative Commons license
cc_models = thingi10k.dataset(license='creative commons')
```

## License

The source code for organize and filter the Thingi10K dataset is licensed under the Apache License,
Version 2.0. Each "thing" in the dataset is licensed under different licenses. Please refer to the
`license` field associated with each entry in the dataset.


