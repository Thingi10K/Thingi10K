""" Thingi10k dataset.


## Usage:

    ```python
    import thingi10k

    thingi10k.init()

    for file_id in thingi10k.file_ids():
        vertices, faces = thingi10k.load_file(file_id)
        # Do something with vertices and faces

    for file_id in thingi10k.filter(thing_ids=[10955]).file_ids()
        vertices, faces = thingi10k.load_file(file_id)
        # Do something with vertices and faces
    ```

"""

from .Thingi10K import __version__

from ._utils import load_file, file_ids, filter, init, dataset
