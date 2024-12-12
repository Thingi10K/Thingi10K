from pathlib import Path
import numpy as np
import numpy.typing as npt
import pandas as pd
import datasets  # type: ignore

from ._logger import logger

root = Path(__file__).parent
metadata_dir = root / "metadata"
_dataset = None


def dataset(
    file_ids: int | npt.ArrayLike | None = None,
    num_vertices: int | None | tuple[int | None, int | None] = None,
    num_facets: int | None | tuple[int | None, int | None] = None,
):
    """Get the (filtered) dataset.

    :param file_ids:     Filter by file ids.
    :param num_vertices: Filter by the number of vertices. If a tuple is provided, it is interpreted
                         as a range. If any of the lower or upper bound is None, it is not
                         considered in the filter.
    :param num_facets:   Filter by the number of facets. If a tuple is provided, it is interpreted
                         as a range. If any of the lower or upper bound is None, it is not
                         considered in the filter.

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

    return d


def input_summary():
    summary = pd.read_csv(metadata_dir / "input_summary.csv")
    return summary


def geometry_data():
    data = pd.read_csv(metadata_dir / "geometry_data.csv")
    return data


def contextual_data():
    data = pd.read_csv(metadata_dir / "contextual_data.csv")
    return data


def tag_data():
    data = pd.read_csv(metadata_dir / "tag_data.csv")
    return data


def file_ids() -> npt.ArrayLike:
    summary = input_summary()
    return summary["ID"].values


def __iter__():
    assert _dataset is not None, "Dataset is not initialized. Call init() first."
    for entry in _dataset["train"]:
        file_id = entry["file_id"]
        file_path = entry["file_path"]
        with np.load(file_path) as data:
            vertices, facets = data["vertices"], data["facets"]
            yield file_id, vertices, facets


class Metadata:
    def __init__(self):
        self.summary = input_summary()
        self.geometry = geometry_data()
        self.contextual = contextual_data()
        self.tags = tag_data()

        self.summary["Closed"] = self.summary["Closed"].astype(bool)
        self.summary["Edge manifold"] = self.summary["Edge manifold"].astype(bool)
        self.summary["Vertex manifold"] = self.summary["Vertex manifold"].astype(bool)
        self.summary["No degenerate faces"] = self.summary[
            "No degenerate faces"
        ].astype(bool)
        self.summary["PWN"] = self.summary["PWN"].astype(bool)

        self.__file_ids = np.array(self.summary["ID"].values)

    def filter(
        self,
        thing_ids: int | npt.ArrayLike | None = None,
        licenses: str | npt.ArrayLike | None = None,
        is_closed: bool | None = None,
        is_edge_manifold: bool | None = None,
        is_vertex_manifold: bool | None = None,
        is_manifold: bool | None = None,
        is_pwn: bool | None = None,
        is_solid: bool | None = None,
    ):
        # Filter by thing_ids
        if thing_ids is not None:
            if isinstance(thing_ids, int):
                thing_ids = [thing_ids]
            selected_file_ids = self.summary[self.summary["Thing ID"].isin(thing_ids)][
                "ID"
            ].values
            self.__file_ids = np.intersect1d(selected_file_ids, self.__file_ids)

        # Filter by licenses
        if licenses is not None:
            if isinstance(licenses, str):
                licenses = [licenses]
            assert isinstance(licenses, list)
            selected_file_ids = self.summary[
                self.summary["License"].str.contains("|".join(licenses), case=False)
            ]["ID"].values
            self.__file_ids = np.intersect1d(selected_file_ids, self.__file_ids)

        # Filter by closed meshes
        if is_closed is not None:
            selected_file_ids = self.summary[self.summary["Closed"] == is_closed][
                "ID"
            ].values
            self.__file_ids = np.intersect1d(selected_file_ids, self.__file_ids)

        # Filter by manifold meshes
        if is_manifold is not None:
            is_vertex_manifold = is_manifold

        # Filter by edge manifold meshes
        if is_edge_manifold is not None:
            selected_file_ids = self.summary[
                self.summary["Edge manifold"] == is_edge_manifold
            ]["ID"].values
            self.__file_ids = np.intersect1d(selected_file_ids, self.__file_ids)

        # Filter by vertex manifold meshes
        if is_vertex_manifold is not None:
            selected_file_ids = self.summary[
                self.summary["Vertex manifold"] == is_vertex_manifold
            ]["ID"].values
            self.__file_ids = np.intersect1d(selected_file_ids, self.__file_ids)

        # Filter by pwn (Piecewise-constant Winding Number) meshes
        if is_pwn is not None:
            selected_file_ids = self.summary[self.summary["PWN"] == is_pwn]["ID"].values
            self.__file_ids = np.intersect1d(selected_file_ids, self.__file_ids)

        # Filter by solid meshes
        if is_solid is not None:
            target_value = 1 if is_solid else 0
            selected_file_ids = self.geometry[self.geometry["solid"] == target_value][
                "file_id"
            ].values
            self.__file_ids = np.intersect1d(selected_file_ids, self.__file_ids)

    def file_ids(self) -> npt.ArrayLike:
        return self.__file_ids


def filter(*args, **kwargs):
    metadata = Metadata()
    metadata.filter(*args, **kwargs)
    return metadata


def load_file(file_id: int) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Load mesh from Thingi10k dataset.

    :param file_id: file id of the mesh.

    :return: vertices and facets of the mesh object."""

    assert _dataset is not None, "Dataset is not initialized. Call init() first."

    if file_id in [49911, 74463, 286163, 77942]:
        logger.warning(
            f"Model {file_id} is known to be corrupted. Returning empty arrays."
        )
        return np.array([], dtype=np.float64), np.array([], dtype=np.int32)

    subdataset = _dataset.filter(lambda x: x["file_id"] == file_id)
    assert len(subdataset["train"]) == 1, f"File {file_id} not found in the dataset."

    mesh_file = subdataset["train"][0]["file_path"]
    with np.load(mesh_file) as data:
        return data["vertices"], data["facets"]


def init():
    global _dataset
    _dataset = datasets.load_dataset(
        str((root / "Thingi10K.py").resolve()), trust_remote_code=True
    )
