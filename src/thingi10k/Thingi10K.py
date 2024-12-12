"""Thingi10K: A Dataset of 10,000 3D-Printing Models"""

import numpy as np
import pathlib
import datasets
import polars as pl

__version__ = "1.1.0"

_corrupt_file_ids = [49911, 74463, 286163, 77942]


_CITATION = """\
@article{Thingi10K,
  title={Thingi10K: A Dataset of 10,000 3D-Printing Models},
  author={Zhou, Qingnan and Jacobson, Alec},
  journal={arXiv preprint arXiv:1605.04797},
  year={2016}
}
"""

_DESCRIPTION = """\
Thingi10K is a large scale 3D dataset created to study the variety, complexity and quality of
real-world 3D printing models. We analyze every mesh of all things featured on Thingiverse.com
between Sept. 16, 2009 and Nov. 15, 2015. On this site, we hope to share our findings with you.
"""

_HOMEPAGE = "https://ten-thousand-models.appspot.com"

_LICENSE = ""


class Thingi10K(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version(__version__)

    def _info(self):
        features = datasets.Features(
            {
                "file_id": datasets.Value("int32"),
                "file_path": datasets.Value("string"),
                "num_vertices": datasets.Value("int32"),
                "num_facets": datasets.Value("int32"),
                "num_components": datasets.Value("int32"),
                "closed": datasets.Value("bool"),
                "self_intersecting": datasets.Value("bool"),
                "vertex_manifold": datasets.Value("bool"),
                "edge_manifold": datasets.Value("bool"),
                "oriented": datasets.Value("bool"),
                "PWN": datasets.Value("bool"),
                "solid": datasets.Value("bool"),
                "euler": datasets.Value("int32"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """
        Define splits and specify where the data files are located.
        """
        repo_url = "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main"
        npz_url = f"{repo_url}/npz"

        metadata_url = "metadata"

        geometry_csv = dl_manager.download(f"{metadata_url}/geometry_data.csv")
        assert pathlib.Path(geometry_csv).exists()

        schema = {
            "file_id": pl.Int32,
            "num_vertices": pl.Int32,
            "num_faces": pl.Int32,
            "num_geometrical_degenerated_faces": pl.Int32,
            "num_combinatorial_degenerated_faces": pl.Int32,
            "num_connected_components": pl.Int32,
            "num_boundary_edges": pl.Int32,
            "num_duplicated_faces": pl.Int32,
            "euler_characteristic": pl.Int32,
            "num_self_intersections": pl.Int32,
            "num_coplanar_intersecting_faces": pl.Int32,
            "vertex_manifold": pl.Int32,
            "edge_manifold": pl.Int32,
            "oriented": pl.Int32,
            "total_area": pl.Float64,
            "min_area": pl.Float64,
            "p25_area": pl.Float64,
            "median_area": pl.Float64,
            "p75_area": pl.Float64,
            "p90_area": pl.Float64,
            "p95_area": pl.Float64,
            "max_area": pl.Float64,
            "min_valance": pl.Int32,
            "p25_valance": pl.Int32,
            "median_valance": pl.Int32,
            "p75_valance": pl.Int32,
            "p90_valance": pl.Int32,
            "p95_valance": pl.Int32,
            "max_valance": pl.Int32,
            "min_dihedral_angle": pl.Float64,
            "p25_dihedral_angle": pl.Float64,
            "median_dihedral_angle": pl.Float64,
            "p75_dihedral_angle": pl.Float64,
            "p90_dihedral_angle": pl.Float64,
            "p95_dihedral_angle": pl.Float64,
            "max_dihedral_angle": pl.Float64,
            "min_aspect_ratio": pl.Float64,
            "p25_aspect_ratio": pl.Float64,
            "median_aspect_ratio": pl.Float64,
            "p75_aspect_ratio": pl.Float64,
            "p90_aspect_ratio": pl.Float64,
            "p95_aspect_ratio": pl.Float64,
            "max_aspect_ratio": pl.Float64,
            "PWN": pl.Int32,
            "solid": pl.Int32,
            "ave_area": pl.Float64,
            "ave_valance": pl.Float64,
            "ave_dihedral_angle": pl.Float64,
            "ave_aspect_ratio": pl.Float64,
        }

        geometry_data = pl.read_csv(
            geometry_csv, schema_overrides=schema, ignore_errors=True
        )
        file_ids = geometry_data["file_id"]

        downloaded_files = dl_manager.download(
            [
                f"{npz_url}/{file_id}.npz"
                for file_id in file_ids
                if file_id not in _corrupt_file_ids
            ]
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "npz_files": downloaded_files,
                    "geometry_data": geometry_data,
                },
            ),
        ]

    def _generate_examples(self, npz_files, geometry_data):
        """
        Yield examples from the .npz files in the specified folder.
        """

        df = geometry_data

        for idx, file_name in enumerate(npz_files):
            file_id = pathlib.Path(file_name).stem

            metadata = df.row(idx, named=True)

            # Yield the data, including the filename
            yield idx, {
                "file_id": int(file_id),
                "file_path": file_name,
                "num_vertices": metadata["num_vertices"],
                "num_facets": metadata["num_faces"],
                "num_components": metadata["num_connected_components"],
                "closed": metadata["num_boundary_edges"] == 0,
                "self_intersecting": metadata["num_self_intersections"] > 0,
                "vertex_manifold": metadata["vertex_manifold"] == 1,
                "edge_manifold": metadata["edge_manifold"] == 1,
                "oriented": metadata["oriented"] == 1,
                "PWN": metadata["PWN"] == 1,
                "solid": metadata["solid"] == 1,
                "euler": metadata["euler_characteristic"],
            }
