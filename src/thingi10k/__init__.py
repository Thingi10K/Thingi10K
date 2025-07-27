"""Thingi10k: A dataset of 10,000 3D-printable models

This package provides a Python interface to the Thingi10k dataset, a collection of 10,000
3D-printable models with geometric and contextual metadata.

## Features

* **Parallel download** with caching support
* **Easy access** to 3D geometry from the Thingi10k dataset
* **Advanced filtering** based on geometric and contextual properties
* **Multiple variants** of the dataset (npz, raw)
* **Semantic search** using CLIP model (optional)

## Quick Start

Download the dataset and iterate over models:

    >>> import thingi10k
    >>> thingi10k.init()  # Download dataset with caching
    >>> for entry in thingi10k.dataset():
    ...     file_id = entry['file_id']
    ...     author = entry['author']
    ...     license = entry['license']
    ...     vertices, facets = thingi10k.load_file(entry['file_path'])
    ...     # vertices: np.ndarray of shape (N, 3) - vertex coordinates
    ...     # facets: np.ndarray of shape (M, 3) - triangle face indices

## Filtering Examples

Filter by geometric properties:

    >>> # Closed meshes with at most 1000 vertices
    >>> for entry in thingi10k.dataset(num_vertices=(None, 1000), closed=True):
    ...     vertices, facets = thingi10k.load_file(entry['file_path'])

    >>> # Solid models with exactly one component
    >>> for entry in thingi10k.dataset(solid=True, num_components=1):
    ...     vertices, facets = thingi10k.load_file(entry['file_path'])

Filter by contextual properties:

    >>> # Models with Creative Commons license
    >>> for entry in thingi10k.dataset(license='creative commons'):
    ...     vertices, facets = thingi10k.load_file(entry['file_path'])

    >>> # Models by specific authors
    >>> for entry in thingi10k.dataset(author=['user1', 'user2']):
    ...     vertices, facets = thingi10k.load_file(entry['file_path'])

    >>> # Models in specific categories with tags
    >>> for entry in thingi10k.dataset(category='Household', tags='kitchen'):
    ...     vertices, facets = thingi10k.load_file(entry['file_path'])

Combine multiple filters:

    >>> # Complex filtering example
    >>> for entry in thingi10k.dataset(
    ...     solid=True,
    ...     num_components=1,
    ...     license='creative commons',
    ...     num_vertices=(100, 5000)
    ... ):
    ...     file_id = entry['file_id']
    ...     vertices, facets = thingi10k.load_file(entry['file_path'])

## Available Filters

**Contextual filters:**
- `file_id`, `thing_id`: Specific model IDs (int or list)
- `author`: Author name(s) (str or list)
- `license`: License text (str or list)
- `category`, `subcategory`: Model categories (str or list)
- `name`: Model name (str or list)
- `tags`: Tags (str or list)
- `query`: Text search across multiple fields (str)

**Geometric filters (support ranges as tuples):**
- `num_vertices`, `num_facets`: Mesh complexity
- `num_components`: Connected components count
- `num_boundary_edges`: Open boundary count
- `euler`, `genus`: Topological properties

**Boolean geometric properties:**
- `closed`: No boundary edges
- `manifold`: Both vertex and edge manifolds
- `vertex_manifold`, `edge_manifold`: Manifold at vertices/edges
- `oriented`: Consistent face orientation
- `self_intersecting`: Has self-intersections
- `pwn`: Induces piecewise-constant winding number field
- `solid`: Closed and oriented

## Dataset Variants

Two variants are available:

**NPZ variant (default):** Pre-extracted geometry in NumPy format - faster download and loading:

    >>> thingi10k.init(variant='npz')  # Default

**Raw variant:** Original mesh files (STL, OBJ, etc.) - slower but preserves original format:

    >>> thingi10k.init(variant='raw')

## Caching

Dataset is cached locally by default:

    >>> # Use default cache location
    >>> thingi10k.init()

    >>> # Specify custom cache directory
    >>> thingi10k.init(cache_dir='path/to/cache')

    >>> # Force re-download
    >>> thingi10k.init(force_redownload=True)

## CLIP-based Semantic Search

For semantic search using natural language queries (requires optional dependencies):

    >>> # To ensure optional dependencies are installed
    >>> # pip install thingi10k[clip]

    >>> import thingi10k
    >>>
    >>> thingi10k.init()
    >>> for entry in thingi10k.dataset(query="cute animal"):
    ...     vertices, facets = thingi10k.load_file(entry['file_path'])

Note that semantic search can be combined with other filters.

"""

__version__ = "1.3.0"

from ._utils import (
    load_file,
    init,
    dataset,
)
from ._logging import logger

__all__ = [
    "dataset",
    "init",
    "load_file",
    "logger",
]
