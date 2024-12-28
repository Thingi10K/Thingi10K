from pathlib import Path
import numpy as np
import numpy.typing as npt
import datasets  # type: ignore
import re
import lagrange
from typing import Literal

root = Path(__file__).parent
_dataset = None


def dataset(
    file_id: int | npt.ArrayLike | None = None,
    thing_id: int | npt.ArrayLike | None = None,
    author: str | npt.ArrayLike | None = None,
    license: str | npt.ArrayLike | None = None,
    category: str | npt.ArrayLike | None = None,
    subcategory: str | npt.ArrayLike | None = None,
    name: str | npt.ArrayLike | None = None,
    tags: str | npt.ArrayLike | None = None,
    num_vertices: int | None | tuple[int | None, int | None] = None,
    num_facets: int | None | tuple[int | None, int | None] = None,
    num_components: int | None | tuple[int | None, int | None] = None,
    num_boundary_edges: int | None | tuple[int | None, int | None] = None,
    closed: bool | None = None,
    self_intersecting: bool | None = None,
    manifold: bool | None = None,
    vertex_manifold: bool | None = None,
    edge_manifold: bool | None = None,
    oriented: bool | None = None,
    pwn: bool | None = None,
    solid: bool | None = None,
    euler: int | None | tuple[int | None, int | None] = None,
    genus: int | None | tuple[int | None, int | None] = None,
) -> datasets.Dataset:
    """Get the (filtered) dataset.

    :param file_id:      Filter by file ids. If an array is provided, match any of the values.
    :param thing_id:     Filter by thing ids. If an array is provided, match any of the values.
    :param author:       Filter by author. If an array is provided, match any of the values.
    :param license:      Filter by license. If an array is provided, match any of the values.
    :param category:     Filter by category. If an array is provided, match any of the values.
    :param subcategory:  Filter by subcategory. If an array is provided, match any of the values.
    :param name:         Filter by model name. If an array is provided, match any of the values.
    :param tags:         Filter by tags. If an array is provided, match any of the values.
    :param num_vertices: Filter by the number of vertices. If a tuple is provided, it is interpreted
                         as a range. If any of the lower or upper bound is None, it is not
                         considered in the filter.
    :param num_facets:   Filter by the number of facets. If a tuple is provided, it is interpreted
                         as a range. If any of the lower or upper bound is None, it is not
                         considered in the filter.
    :param num_components: Filter by the number of connected components. If a tuple is provided, it
                           is interpreted as a range. If any of the lower or upper bound is None, it
                           is not considered in the filter.
    :param num_boundary_edges: Filter by the number of boundary edges. If a tuple is provided, it is
                               interpreted as a range. If any of the lower or upper bound is None, it
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
    :param genus:        Filter by the genus. If a tuple is provided, it is interpreted as a range.
                         If any of the lower or upper bound is None, it is not considered in the
                         filter.

    :returns: The filtered dataset.
    """
    assert _dataset is not None, "Dataset is not initialized. Call init() first."
    d = _dataset["train"]

    if file_id is not None:
        if isinstance(file_id, int):
            file_id = [file_id]
        d = d.filter(lambda x: x["file_id"] in file_id)

    if thing_id is not None:
        if isinstance(thing_id, int):
            thing_id = [thing_id]
        d = d.filter(lambda x: x["thing_id"] in thing_id)

    if author is not None:
        if isinstance(author, str):
            author = [author]
        d = d.filter(lambda x: x["author"] in author)

    if license is not None:
        if isinstance(license, str):
            license = [license]
        d = d.filter(
            lambda x: any(
                re.search(lic, x["license"], re.IGNORECASE) for lic in license
            )
        )

    if category is not None:
        if isinstance(category, str):
            category = [category]
        d = d.filter(
            lambda x: any(
                re.search(entry, x["category"], re.IGNORECASE) for entry in category
            )
        )

    if subcategory is not None:
        if isinstance(subcategory, str):
            subcategory = [subcategory]
        d = d.filter(
            lambda x: any(
                re.search(entry, x["subcategory"], re.IGNORECASE)
                for entry in subcategory
            )
        )

    if name is not None:
        if isinstance(name, str):
            name = [name]
        d = d.filter(
            lambda x: any(re.search(entry, x["name"], re.IGNORECASE) for entry in name)
        )

    if tags is not None:
        if isinstance(tags, str):
            tags = [tags]
        d = d.filter(
            lambda x: any(
                re.search(entry, ",".join(x["tags"]), re.IGNORECASE) for entry in tags
            )
        )

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

    if num_boundary_edges is not None:
        if isinstance(num_boundary_edges, int):
            num_boundary_edges = (num_boundary_edges, num_boundary_edges)
        assert isinstance(num_boundary_edges, tuple)
        assert len(num_boundary_edges) == 2
        if num_boundary_edges[0] is not None:
            d = d.filter(lambda x: x["num_boundary_edges"] >= num_boundary_edges[0])
        if num_boundary_edges[1] is not None:
            d = d.filter(lambda x: x["num_boundary_edges"] <= num_boundary_edges[1])

    if closed is not None:
        d = d.filter(lambda x: x["closed"] == closed)

    if self_intersecting is not None:
        d = d.filter(lambda x: x["self_intersecting"] == self_intersecting)

    if manifold is not None:
        vertex_manifold = edge_manifold = manifold

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

    if genus is not None:
        if isinstance(genus, int):
            genus = (genus, genus)
        assert isinstance(genus, tuple)
        assert len(genus) == 2
        if genus[0] is not None:
            d = d.filter(
                lambda x: x["num_components"] == 1
                and x["num_boundary_edges"] == 0
                and x["vertex_manifold"]
                and x["euler"] % 2 == 0
                and (2 - x["euler"]) // 2 >= genus[0]
            )
        if genus[1] is not None:
            d = d.filter(
                lambda x: x["num_components"] == 1
                and x["num_boundary_edges"] == 0
                and x["vertex_manifold"]
                and x["euler"] % 2 == 0
                and (2 - x["euler"]) // 2 <= genus[1]
            )

    return d


def load_file(file_path: str) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Load the vertices and facets from a file.

    :param file_path: The path to the file.

    :returns: The vertices and facets.
    """
    if Path(file_path).suffix == ".npz":
        # Unpack npz file.
        with np.load(file_path) as data:
            return data["vertices"], data["facets"]
    else:
        # Load raw mesh file with lagrange.
        mesh = lagrange.io.load_mesh(file_path)
        return mesh.vertices, mesh.facets


def init(
    variant: Literal["npz", "raw"] | None = None,
    cache_dir: str | None = None,
    force_redownload: bool = False,
):
    """Initialize the dataset.

    :param variant:          The variant of the dataset to load. Options are "npz" and "raw".
                             Default is "npz".
    :param cache_dir:        The directory where the dataset is cached.
    :param force_redownload: Whether to force redownload the dataset.
    """
    global _dataset
    if force_redownload:
        download_mode = "force_redownload"
    else:
        download_mode = "reuse_dataset_if_exists"

    _dataset = datasets.load_dataset(
        "Thingi10K/Thingi10K",
        trust_remote_code=True,
        cache_dir=cache_dir,
        download_mode=download_mode,
        name=variant,
    )
