""" Thingi10k: A dataset of 10,000 3D-printable models

This package provides a Python interface to the Thingi10k dataset.

Features:
    - Parallel download dataset with caching support
    - Easy access of 3D geometry from the Thingi10k dataset
    - Filter dataset based on geometric and contextual properties
    - Support different variants of the dataset

Usage:

    Download the dataset using `thingi10k.init()` function:

    >>> import thingi10k
    >>> thingi10k.init()

   Iterate over the dataset and extract 3D geometry:

    >>> for entry in thingi10k.dataset():
    ...     file_id = entry['file_id']
    ...     author = entry['author']
    ...     license = entry['license']
    ...     vertices, facets = thingi10k.load_file(entry['file_path'])

    Filter dataset based on geometric properties:

    >>> for entry in thingi10k.dataset(num_vertices=(None, 1000), closed=True):
    ...     # Iterate over all closed mesh with at most 1000 vertices
    ...     file_id = entry['file_id']
    ...     vertices, facets = thingi10k.load_file(entry['file_path'])

    Filter dataset based on contextual properties:

    >>> for entry in thingi10k.dataset(license='creative commons'):
    ...     # Iterate over all models with license containing 'creative commons'
    ...     file_id = entry['file_id']
    ...     vertices, facets = thingi10k.load_file(entry['file_path'])

    Filter dataset based on multiple properties:

    >>> for entry in thingi10k.dataset(solid=True, num_components=1, license='creative commons'):
    ...     # Iterate over all solid models with one component and license containing
    ...     # 'creative commons'
    ...     file_id = entry['file_id']
    ...     vertices, facets = thingi10k.load_file(entry['file_path'])

Variants of the dataset:

    There are two variants of the Thingi10K datasets.

    * The `raw` variant contains the raw mesh files in the dataset. It is slower to download and
      requires parsing the mesh files.
    * The `npz` variant contains the extracted geometry in npz format. Is is faster to download and
      does not require parsing the mesh files. This is the default.

    Use the `variant` argument of the `init()` function to load the desired variant:

    >>> thingi10k.init(variant='npz') # Load the npz variant of the dataset (default)
    >>> thingi10k.init(variant='raw') # Load the raw variant of the dataset

Caching:

    The dataset is cached by default. Use the `cache_dir` argument of the `init()` function to
    specify the cache directory:

    >>> thingi10k.init(cache_dir='path/to/cache') # Load the dataset with caching enabled

    To force redownload the dataset, use the `force_redownload` argument of the `init()` function:

    >>> thingi10k.init(force_redownload=True) # Force redownload the dataset

"""

__version__ = '1.1.5'

from ._utils import load_file, init, dataset
