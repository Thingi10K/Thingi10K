""" Thingi10k dataset.


## Usage:

    ```python
    import thingi10k

    thingi10k.init()

    # Iterate over all data
    for entry in thingi10k.dataset():
        file_id = entry['file_id']
        vertices, facets = thingi10k.load_file(entry['file_path'])

    # Iterate over closed mesh with at most 1000 vertices
    for entry in thingi10k.dataset(num_vertices=(None, 1000), closed=True):
        file_id = entry['file_id']
        vertices, facets = thingi10k.load_file(entry['file_path'])
    ```
"""

__version__ = '1.1.3'

from ._utils import load_file, init, dataset
