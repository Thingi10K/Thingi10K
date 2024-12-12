""" Thingi10k dataset.


## Usage:

    ```python
    import thingi10k

    for file_id in thingi10k.file_ids():
        vertices, faces = thingi10k.load_file(file_id)
        # Do something with vertices and faces

    for file_id in thingi10k.filter(thing_ids=[10955]).file_ids()
        vertices, faces = thingi10k.load_file(file_id)
        # Do something with vertices and faces
    ```

"""

__version__ = "0.1.0"

from .utils import load_file, file_ids, filter
