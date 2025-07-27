from pathlib import Path
import numpy as np
import numpy.typing as npt
import datasets  # type: ignore
import re
import lagrange
import logging
from typing import Literal, Union, Any, Sequence
from ._builder import Thingi10KBuilder
from ._clip import with_clip, ClipFeatures
from ._logging import logger


root = Path(__file__).parent
_dataset = None
_clip_features: ClipFeatures | None = None

# Type aliases for better readability
FilterValueType = Union[int, str, bool, None]
ArrayLikeType = Union[npt.ArrayLike, Sequence[Any]]
RangeType = Union[int, None, tuple[int | None, int | None]]


class DatasetFilters:
    """Helper class to organize dataset filtering logic."""

    @staticmethod
    def _normalize_to_list(value: Union[FilterValueType, ArrayLikeType]) -> list:
        """Convert single values or array-like to list."""
        if isinstance(value, (int, str, bool)):
            return [value]
        return list(value) if value is not None else []  # type: ignore[arg-type]

    @staticmethod
    def _normalize_range(value: RangeType) -> tuple[int | None, int | None]:
        """Normalize range input to tuple format."""
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, tuple) and len(value) == 2:
            return value
        if value is None:
            return (None, None)
        raise ValueError(
            f"Invalid range format: {value}. Expected int, tuple of 2 ints, or None."
        )

    @staticmethod
    def apply_exact_filters(dataset: datasets.Dataset, **filters) -> datasets.Dataset:
        """Apply filters for exact matches."""
        d = dataset

        # ID-based filters
        for field in ["file_id", "thing_id"]:
            if filters.get(field) is not None:
                values = DatasetFilters._normalize_to_list(filters[field])
                d = d.filter(lambda x, field=field, values=values: x[field] in values)

        # String-based filters (exact matches)
        if filters.get("author") is not None:
            authors = DatasetFilters._normalize_to_list(filters["author"])
            d = d.filter(lambda x: x["author"] in authors)

        # Boolean filters
        for field in [
            "closed",
            "self_intersecting",
            "vertex_manifold",
            "edge_manifold",
            "oriented",
            "solid",
        ]:
            if filters.get(field) is not None:
                d = d.filter(
                    lambda x, field=field, value=filters[field]: x[field] == value
                )

        # Special case for PWN (different field name)
        if filters.get("pwn") is not None:
            d = d.filter(lambda x: x["PWN"] == filters["pwn"])

        return d

    @staticmethod
    def apply_regex_filters(dataset: datasets.Dataset, **filters) -> datasets.Dataset:
        """Apply filters using regex pattern matching."""
        d = dataset

        # String fields with regex matching
        regex_fields = ["license", "category", "subcategory", "name"]
        for field in regex_fields:
            if filters.get(field) is not None:
                patterns = DatasetFilters._normalize_to_list(filters[field])
                d = d.filter(
                    lambda x, field=field, patterns=patterns: any(
                        re.search(pattern, x[field], re.IGNORECASE)
                        for pattern in patterns
                    )
                )

        # Tags require special handling
        if filters.get("tags") is not None:
            tag_patterns = DatasetFilters._normalize_to_list(filters["tags"])
            d = d.filter(
                lambda x: any(
                    re.search(pattern, ",".join(x["tags"]), re.IGNORECASE)
                    for pattern in tag_patterns
                )
            )

        return d

    @staticmethod
    def apply_range_filters(dataset: datasets.Dataset, **filters) -> datasets.Dataset:
        """Apply filters for numeric ranges."""
        d = dataset

        range_fields = [
            "num_vertices",
            "num_facets",
            "num_components",
            "num_boundary_edges",
            "euler",
        ]

        for field in range_fields:
            if filters.get(field) is not None:
                try:
                    min_val, max_val = DatasetFilters._normalize_range(filters[field])
                    if min_val is not None:
                        d = d.filter(
                            lambda x, field=field, min_val=min_val: x[field] >= min_val
                        )
                    if max_val is not None:
                        d = d.filter(
                            lambda x, field=field, max_val=max_val: x[field] <= max_val
                        )
                except ValueError as e:
                    logger.warning(f"Invalid range for {field}: {e}")
                    continue

        return d

    @staticmethod
    def apply_genus_filter(
        dataset: datasets.Dataset, genus: RangeType
    ) -> datasets.Dataset:
        """Apply complex genus filter with validation."""
        if genus is None:
            return dataset

        try:
            min_genus, max_genus = DatasetFilters._normalize_range(genus)
        except ValueError as e:
            logger.warning(f"Invalid genus range: {e}")
            return dataset

        d = dataset

        # Common conditions for genus calculation
        def base_conditions(x):
            return (
                x["num_components"] == 1
                and x["num_boundary_edges"] == 0
                and x["vertex_manifold"]
                and x["euler"] % 2 == 0
            )

        if min_genus is not None:
            d = d.filter(
                lambda x: base_conditions(x) and (2 - x["euler"]) // 2 >= min_genus
            )
        if max_genus is not None:
            d = d.filter(
                lambda x: base_conditions(x) and (2 - x["euler"]) // 2 <= max_genus
            )

        return d


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
    query: str | None = None,
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
    :param manifold:     Filter by manifold/non-manifold meshes (sets both vertex and edge manifold).
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
    :param query:        A free-form text query to search for in the dataset. This feature requires
                         CLIP model to be enabled.

    :returns: The filtered dataset.

    :raises RuntimeError: If dataset is not initialized.
    """
    if _dataset is None:
        raise RuntimeError("Dataset is not initialized. Call init() first.")

    # Handle manifold parameter (affects both vertex and edge manifold)
    if manifold is not None:
        vertex_manifold = edge_manifold = manifold

    # Prepare filter arguments
    filter_args = {
        "file_id": file_id,
        "thing_id": thing_id,
        "author": author,
        "license": license,
        "category": category,
        "subcategory": subcategory,
        "name": name,
        "tags": tags,
        "num_vertices": num_vertices,
        "num_facets": num_facets,
        "num_components": num_components,
        "num_boundary_edges": num_boundary_edges,
        "closed": closed,
        "self_intersecting": self_intersecting,
        "vertex_manifold": vertex_manifold,
        "edge_manifold": edge_manifold,
        "oriented": oriented,
        "pwn": pwn,
        "solid": solid,
        "euler": euler,
    }

    # Apply filters in logical groups
    d = _dataset["train"]
    d = DatasetFilters.apply_exact_filters(d, **filter_args)
    d = DatasetFilters.apply_regex_filters(d, **filter_args)
    d = DatasetFilters.apply_range_filters(d, **filter_args)
    d = DatasetFilters.apply_genus_filter(d, genus)

    if query is not None:
        assert with_clip, "CLIP model is not available. Please `pip install thingi10k[clip]`."
        assert _clip_features is not None, "CLIP features are not initialized."
        selected_file_ids = _clip_features.query(query)
        d = d.filter(lambda x: x["file_id"] in selected_file_ids)

    logger.info(f"Filtered dataset from {len(_dataset['train'])} to {len(d)} entries")
    return d


def load_file(
    file_path: str,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
    """Load the vertices and facets from a file.

    :param file_path: The path to the file.
    :returns: The vertices and facets as numpy arrays.

    :raises FileNotFoundError: If the file doesn't exist.
    :raises ValueError: If the file format is unsupported or corrupted.
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if file_path_obj.suffix == ".npz":
            # Unpack npz file
            with np.load(file_path) as data:
                if "vertices" not in data or "facets" not in data:
                    raise ValueError(f"NPZ file missing required arrays: {file_path}")
                vertices = np.asarray(data["vertices"], dtype=np.floating)
                facets = np.asarray(data["facets"], dtype=np.integer)
                return vertices, facets
        else:
            # Load raw mesh file with lagrange
            mesh = lagrange.io.load_mesh(file_path)
            vertices = np.asarray(mesh.vertices, dtype=np.floating)
            facets = np.asarray(mesh.facets, dtype=np.integer)
            return vertices, facets
    except Exception as e:
        raise ValueError(f"Failed to load mesh file {file_path}: {e}") from e


def init(
    variant: Literal["npz", "raw"] | None = None,
    cache_dir: str | None = None,
    force_redownload: bool = False,
) -> None:
    """Initialize the dataset.

    :param variant:          The variant of the dataset to load. Options are "npz" and "raw".
                             Default is "npz".
    :param cache_dir:        The directory where the dataset is cached.
    :param force_redownload: Whether to force redownload the dataset.

    :raises ValueError: If variant is not supported.
    :raises RuntimeError: If dataset initialization fails.
    """
    global _dataset, _clip_features

    if variant is not None and variant not in ["npz", "raw"]:
        raise ValueError(f"Unsupported variant: {variant}. Must be 'npz' or 'raw'.")

    try:
        download_config = datasets.DownloadConfig()
        if cache_dir is not None:
            download_config.cache_dir = cache_dir
        download_config.force_download = force_redownload

        builder = Thingi10KBuilder(config_name=variant)
        builder.download_and_prepare(download_config=download_config)
        _dataset = builder.as_dataset()

        logger.info(
            f"Dataset initialized with {len(_dataset['train'])} entries using variant '{variant or 'npz'}'"
        )

        if with_clip:
            _clip_features = ClipFeatures()

    except Exception as e:
        raise RuntimeError(f"Failed to initialize dataset: {e}") from e
