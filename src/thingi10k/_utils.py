from pathlib import Path
import numpy as np
import numpy.typing as npt
import datasets  # type: ignore

from ._logger import logger

root = Path(__file__).parent
metadata_dir = root / "metadata"
_dataset = None


def dataset(
    file_ids: int | npt.ArrayLike | None = None,
    num_vertices: int | None | tuple[int | None, int | None] = None,
    num_facets: int | None | tuple[int | None, int | None] = None,
    num_components: int | None | tuple[int | None, int | None] = None,
    closed: bool | None = None,
    self_intersecting: bool | None = None,
    vertex_manifold: bool | None = None,
    edge_manifold: bool | None = None,
    oriented: bool | None = None,
    pwn: bool | None = None,
    solid: bool | None = None,
    euler: int | None | tuple[int | None, int | None] = None,
):
    """Get the (filtered) dataset.

    :param file_ids:     Filter by file ids.
    :param num_vertices: Filter by the number of vertices. If a tuple is provided, it is interpreted
                         as a range. If any of the lower or upper bound is None, it is not
                         considered in the filter.
    :param num_facets:   Filter by the number of facets. If a tuple is provided, it is interpreted
                         as a range. If any of the lower or upper bound is None, it is not
                         considered in the filter.
    :param num_components: Filter by the number of connected components. If a tuple is provided, it
                           is interpreted as a range. If any of the lower or upper bound is None, it
                           is not considered in the filter.
    :param closed:       Filter by open/closed meshes.
    :param self_intersecting: Filter by self-intersecting/non-self-intersecting meshes.
    :param vertex_manifold: Filter by vertex-manifold/non-vertex-manifold meshes.
    :param edge_manifold: Filter by edge-manifold/non-edge-manifold meshes.
    :param oriented:     Filter by oriented/non-oriented meshes.
    :param pwn:          Filter by piecewise-constant winding number (PWN) meshes.
    :param solid:        Filter by solid/non-solid meshes.
    :param euler:        Filter by the Euler characteristic. If a tuple is provided, it is
                         interpreted as a range. If any of the lower or upper bound is None, it is
                         not considered in the filter.

    :returns: The filtered dataset.
    """
    assert _dataset is not None, "Dataset is not initialized. Call init() first."
    d = _dataset["train"]

    if file_ids is not None:
        if isinstance(file_ids, int):
            file_ids = [file_ids]
        d = d.filter(lambda x: x["file_id"] in file_ids)

    if num_vertices is not None:
        if isinstance(num_vertices, int):
            num_vertices = (num_vertices, num_vertices)
        assert isinstance(num_vertices, tuple)
        assert len(num_vertices) == 2
        if num_vertices[0] is not None:
            d = d.filter(lambda x: x["num_vertices"] >= num_vertices[0])
        if num_vertices[1] is not None:
            d = d.filter(lambda x: x["num_vertices"] <= num_vertices[1])

    if num_facets is not None:
        if isinstance(num_facets, int):
            num_facets = (num_facets, num_facets)
        assert isinstance(num_facets, tuple)
        assert len(num_facets) == 2
        if num_facets[0] is not None:
            d = d.filter(lambda x: x["num_facets"] >= num_facets[0])
        if num_facets[1] is not None:
            d = d.filter(lambda x: x["num_facets"] <= num_facets[1])

    if num_components is not None:
        if isinstance(num_components, int):
            num_components = (num_components, num_components)
        assert isinstance(num_components, tuple)
        assert len(num_components) == 2
        if num_components[0] is not None:
            d = d.filter(lambda x: x["num_components"] >= num_components[0])
        if num_components[1] is not None:
            d = d.filter(lambda x: x["num_components"] <= num_components[1])

    if closed is not None:
        d = d.filter(lambda x: x["closed"] == closed)

    if self_intersecting is not None:
        d = d.filter(lambda x: x["self_intersecting"] == self_intersecting)

    if vertex_manifold is not None:
        d = d.filter(lambda x: x["vertex_manifold"] == vertex_manifold)

    if edge_manifold is not None:
        d = d.filter(lambda x: x["edge_manifold"] == edge_manifold)

    if oriented is not None:
        d = d.filter(lambda x: x["oriented"] == oriented)

    if pwn is not None:
        d = d.filter(lambda x: x["PWN"] == pwn)

    if solid is not None:
        d = d.filter(lambda x: x["solid"] == solid)

    if euler is not None:
        if isinstance(euler, int):
            euler = (euler, euler)
        assert isinstance(euler, tuple)
        assert len(euler) == 2
        if euler[0] is not None:
            d = d.filter(lambda x: x["euler"] >= euler[0])
        if euler[1] is not None:
            d = d.filter(lambda x: x["euler"] <= euler[1])

    return d


def load_file(file_path: str) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Load the vertices and facets from a file.

    :param file_path: The path to the file.

    :returns: The vertices and facets.
    """
    with np.load(file_path) as data:
        return data["vertices"], data["facets"]


def init(
    cache_dir: str | None = None,
    force_redownload: bool = False,
):
    """ Initialize the dataset.

    :param cache_dir:        The directory where the dataset is cached.
    :param force_redownload: Whether to force redownload the dataset.
    """
    global _dataset
    if force_redownload:
        download_mode = "force_redownload"
    else:
        download_mode = "reuse_dataset_if_exists"

    _dataset = datasets.load_dataset(
        str((root / "Thingi10K.py").resolve()),
        trust_remote_code=True,
        cache_dir=cache_dir,
        download_mode=download_mode,
    )