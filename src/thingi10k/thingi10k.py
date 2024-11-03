from pathlib import Path
import numpy as np
import numpy.typing as npt
import pandas as pd

root = Path(__file__).parent
metadata_dir = root / "metadata"


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

    :return: vertices and faces of the mesh object."""

    mesh_data = root / f"npz/{file_id}.npz"
    data = np.load(mesh_data)
    return data["vertices"], data["faces"]
