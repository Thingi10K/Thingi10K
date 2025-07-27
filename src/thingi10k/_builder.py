"""Thingi10K: A Dataset of 10,000 3D-Printing Models"""

import datasets  # type: ignore
import datetime
import numpy as np
import pathlib
import polars as pl
from typing import Any, Dict, List, Iterator, Tuple
from ._logging import logger


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

_LICENSE = ""  # See license field associated with each model.


class DatasetConfig:
    """Configuration constants for the Thingi10K dataset."""

    REPO_URL = "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main"
    CORRUPT_FILE_IDS = frozenset([49911, 74463, 286163, 77942])

    # Schema definitions
    GEOMETRY_SCHEMA = {
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

    # Default date for missing values
    DEFAULT_DATE = datetime.datetime(1900, 1, 1)


class Thingi10KBuilder(datasets.GeneratorBasedBuilder):
    """
    Thingi10K Dataset builder.
    """

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="npz",
            version="1.0.0",
            description="Dataset stored in .npz format.",
        ),
        datasets.BuilderConfig(
            name="raw",
            version="1.0.0",
            description="Dataset stored in their original raw mesh format.",
        ),
    ]

    DEFAULT_CONFIG_NAME = (
        "npz"  # Faster to download and load, no mesh format parsing needed.
    )

    def _info(self):
        """
        Define the dataset (column) information.
        """
        features = datasets.Features(
            {
                "file_id": datasets.Value("int32"),
                "thing_id": datasets.Value("int32"),
                "file_path": datasets.Value("string"),
                "author": datasets.Value("string"),
                "date": datasets.Value("date64"),
                "license": datasets.Value("string"),
                "category": datasets.Value("string"),
                "subcategory": datasets.Value("string"),
                "name": datasets.Value("string"),
                "tags": datasets.Sequence(datasets.Value("string")),
                "num_vertices": datasets.Value("int32"),
                "num_facets": datasets.Value("int32"),
                "num_components": datasets.Value("int32"),
                "num_boundary_edges": datasets.Value("int32"),
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
        csv_files = self._download_metadata_files(dl_manager)
        dataframes = self._load_and_process_csv_files(csv_files)
        downloaded_files = self._prepare_dataset_files(
            dl_manager, dataframes["geometry_data"], dataframes["summary_data"]
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"downloaded_files": downloaded_files, **dataframes},
            )
        ]

    def _download_metadata_files(self, dl_manager) -> dict[str, str]:
        """Download all required CSV metadata files."""
        metadata_url = f"{DatasetConfig.REPO_URL}/metadata"

        files = {}
        for file_type in [
            "geometry_data",
            "contextual_data",
            "input_summary",
            "tag_data",
        ]:
            url = f"{metadata_url}/{file_type}.csv"
            files[file_type] = dl_manager.download(url)

            # Validate file exists
            if not pathlib.Path(files[file_type]).exists():
                raise FileNotFoundError(f"Failed to download {file_type}.csv")

        return files

    def _load_and_process_csv_files(
        self, csv_files: dict[str, str]
    ) -> dict[str, pl.DataFrame]:
        """Load and process CSV files into Polars DataFrames."""
        dataframes = {}
        schema: Dict[str, Any] = {}
        for file_type, file_path in csv_files.items():
            if file_type == "geometry_data":
                schema = DatasetConfig.GEOMETRY_SCHEMA
                dataframes["geometry_data"] = pl.read_csv(
                    file_path, schema_overrides=schema, ignore_errors=True
                )
            elif file_type == "contextual_data":
                schema = {
                    "Thing ID": pl.Int32,
                    "Date": pl.Datetime,
                    "Category": pl.String,
                    "Sub-category": pl.String,
                    "Name": pl.String,
                    "Author": pl.String,
                    "License": pl.String,
                }
                dataframes["contextual_data"] = pl.read_csv(
                    file_path, schema_overrides=schema, ignore_errors=True
                )
            elif file_type == "input_summary":
                schema = {
                    "ID": pl.Int32,
                    "Thing ID": pl.Int32,
                }
                dataframes["summary_data"] = pl.read_csv(
                    file_path, schema_overrides=schema, ignore_errors=True
                )
            elif file_type == "tag_data":
                schema = {
                    "Thing ID": pl.Int32,
                    "Tag": pl.String,
                }
                dataframes["tag_data"] = pl.read_csv(
                    file_path, schema_overrides=schema, ignore_errors=True
                )
        return dataframes

    def _prepare_dataset_files(
        self, dl_manager, geometry_data: pl.DataFrame, summary_data: pl.DataFrame
    ) -> list[pathlib.Path]:
        """Prepare the dataset files for download."""
        repo_url = DatasetConfig.REPO_URL

        file_ids = geometry_data["file_id"]

        if self.config.name == "npz":
            extraction_dir = dl_manager.download_and_extract(
                f"{repo_url}/Thingi10K_npz.tar.gz"
            )
            extraction_dir = pathlib.Path(extraction_dir)
            if not extraction_dir.exists() or not extraction_dir.is_dir():
                raise FileNotFoundError(
                    f"Extraction directory not found: {extraction_dir}"
                )
            downloaded_files = [
                extraction_dir / "npz" / f"{file_id}.npz"
                for file_id in file_ids
                if file_id not in DatasetConfig.CORRUPT_FILE_IDS
            ]
        elif self.config.name == "raw":
            raw_data = summary_data.select(["ID", "Link"])
            extraction_dir = dl_manager.download_and_extract(
                f"{repo_url}/Thingi10K.tar.gz"
            )
            extraction_dir = pathlib.Path(extraction_dir)
            if not extraction_dir.exists() or not extraction_dir.is_dir():
                raise FileNotFoundError(
                    f"Extraction directory not found: {extraction_dir}"
                )
            downloaded_files = [
                extraction_dir
                / "Thingi10K"
                / "raw_meshes"
                / f"{row[0]}.{row[1].split('.')[-1].lower()}"
                for row in raw_data.iter_rows()
                if row[0] not in DatasetConfig.CORRUPT_FILE_IDS
            ]
        else:
            raise ValueError(f"Unknown config name: {self.config.name}")

        return downloaded_files

    def _prepare_dataframe(
        self,
        geometry_data: pl.DataFrame,
        contextual_data: pl.DataFrame,
        summary_data: pl.DataFrame,
        tag_data: pl.DataFrame,
    ) -> pl.DataFrame:
        """Prepare and join all dataframes efficiently."""

        # Start with geometry data
        df = geometry_data

        # Join with summary data (thing file IDs)
        if summary_data is not None:
            df = df.join(summary_data, left_on="file_id", right_on="ID", how="left")

        # Join with contextual data
        if contextual_data is not None:
            df = df.join(contextual_data, on="Thing ID", how="left")

        # Pre-process and join tag data
        if tag_data is not None:
            tag_data_agg = tag_data.group_by("Thing ID").agg(
                pl.col("Tag").alias("Tags")
            )
            df = df.join(tag_data_agg, on="Thing ID", how="left")

        # Fill nulls in one operation (only if columns exist)
        fill_expressions = []
        if "License" in df.columns:
            fill_expressions.append(pl.col("License").fill_null("unknown"))
        if "Author" in df.columns:
            fill_expressions.append(pl.col("Author").fill_null("unknown"))
        if "Date" in df.columns:
            fill_expressions.append(
                pl.col("Date").fill_null(DatasetConfig.DEFAULT_DATE)
            )
        if "Category" in df.columns:
            fill_expressions.append(pl.col("Category").fill_null("unknown"))
        if "Sub-category" in df.columns:
            fill_expressions.append(pl.col("Sub-category").fill_null("unknown"))
        if "Name" in df.columns:
            fill_expressions.append(pl.col("Name").fill_null("unknown"))
        if "Tags" in df.columns:
            fill_expressions.append(pl.col("Tags").fill_null(pl.lit([])))

        if fill_expressions:
            df = df.with_columns(fill_expressions)

        return df

    def _generate_examples(
        self,
        downloaded_files: List[str],
        geometry_data: pl.DataFrame,
        contextual_data: pl.DataFrame,
        summary_data: pl.DataFrame,
        tag_data: pl.DataFrame,
    ) -> Iterator[Tuple[int, Dict]]:
        """Generate dataset examples with proper typing."""

        # Prepare dataframe once
        df = self._prepare_dataframe(
            geometry_data, contextual_data, summary_data, tag_data
        )

        # Create a dictionary for O(1) lookups
        metadata_dict = {row["file_id"]: row for row in df.iter_rows(named=True)}

        for idx, file_path in enumerate(downloaded_files):
            if not self._is_file_valid(pathlib.Path(file_path)):
                continue

            file_id = int(pathlib.Path(file_path).stem)

            if file_id not in metadata_dict:
                logger.warning(f"No metadata found for file_id: {file_id}")
                continue

            metadata = metadata_dict[file_id]

            # Yield the data, including the filename
            yield idx, {
                "file_id": int(file_id),
                "thing_id": metadata["Thing ID"],
                "file_path": file_path,
                "author": metadata["Author"],
                "date": metadata["Date"],
                "license": metadata["License"],
                "category": metadata["Category"],
                "subcategory": metadata["Sub-category"],
                "name": metadata["Name"],
                "tags": metadata["Tags"],
                "num_vertices": metadata["num_vertices"],
                "num_facets": metadata["num_faces"],
                "num_components": metadata["num_connected_components"],
                "num_boundary_edges": metadata["num_boundary_edges"],
                "closed": metadata["num_boundary_edges"] == 0,
                "self_intersecting": metadata["num_self_intersections"] > 0,
                "vertex_manifold": metadata["vertex_manifold"] == 1,
                "edge_manifold": metadata["edge_manifold"] == 1,
                "oriented": metadata["oriented"] == 1,
                "PWN": metadata["PWN"] == 1,
                "solid": metadata["solid"] == 1,
                "euler": metadata["euler_characteristic"],
            }

    def _is_file_valid(self, file_path: pathlib.Path) -> bool:
        """Check if a file exists and is not corrupted."""
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return False

        if file_path.stat().st_size == 0:
            logger.warning(f"Empty file: {file_path}")
            return False

        return True
