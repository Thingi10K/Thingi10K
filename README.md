## Thingi10K Dataset

Thingi10K is a large scale 3D dataset created to study the variety, complexity and quality of
real-world 3D printing models. We analyze every mesh of all things featured on Thingiverse.com
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

## Usage

```py
import thingi10k

# Get file_ids of all things
file_ids = thingi10k.file_ids()

# Get geometry (i.e. vertices, faces) of a given `file_id`
for file_id in file_ids:
    V, F = thingi10k.load_file(file_id)
```

To filter the dataset based on geometric or contextual properties, use the `filter` function.

```py
import thingi10k

# Filter the dataset based on piecewise-constant winding number (PWN)
pwn_file_ids = thingi10k.filter(is_pwn=True)

# Filter the dataset based on solidness
solid_file_ids = thingi10k.filter(is_solid=True)
```

## License

The source code for organize and filter the Thingi10K dataset is licensed under the Apache License,
Version 2.0. Each "thing" in the dataset is licensed under different licenses. Please refer to the
...


