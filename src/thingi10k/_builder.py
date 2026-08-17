"""Thingi10K: A Dataset of 10,000 3D-Printing Models"""

import datasets  # type: ignore
import datetime
import os
import pathlib
import polars as pl
import shutil
import tarfile
from filelock import FileLock
from huggingface_hub import get_hf_file_metadata
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
Thingi10K is a large-scale 3D dataset created to study the variety, complexity and quality of
real-world 3D printing models. We analyze every mesh of all things featured on Thingiverse.com
between Sept. 16, 2009 and Nov. 15, 2015. On this site, we hope to share our findings with you.
"""

_HOMEPAGE = "https://ten-thousand-models.appspot.com"

_LICENSE = ""  # See license field associated with each model.


class DatasetConfig:
    """Configuration constants for the Thingi10K dataset."""

    # Pin to an immutable Hub tag so downloads are reproducible and a moving
    # `main` never silently changes the data. Bump this when publishing an
    # updated dataset revision.
    REVISION = "v1.5.0"
    REPO_URL = (
        f"https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/{REVISION}"
    )
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

    # Per-variant single-file archive and the top-level directory expected
    # inside it (used as a sanity check after extraction).
    ARCHIVES = {
        "npz": ("Thingi10K_npz.tar.gz", "npz"),
        "raw": ("Thingi10K.tar.gz", "Thingi10K"),
        "tetwild": ("Thingi10K_tetwild_npz.tar.gz", "tetwild"),
    }


def _variant_extract_dir(download_config, variant: str) -> pathlib.Path:
    """Stable directory a variant's archive is extracted into."""
    cache_dir = getattr(download_config, "cache_dir", None)
    base = (
        pathlib.Path(cache_dir)
        if cache_dir
        else pathlib.Path(datasets.config.HF_DATASETS_CACHE)
    )
    return base / f"thingi10k_{variant}_extracted"


def _variant_lock_path(extract_dir: pathlib.Path) -> pathlib.Path:
    """Lock file serializing extraction/clearing of ``extract_dir``."""
    return extract_dir.parent / f"{extract_dir.name}.lock"


def _extract_archive(archive: str, extract_dir: pathlib.Path) -> None:
    """Extract a dataset tar.gz into ``extract_dir``, safely on any Python."""
    with tarfile.open(archive, "r:gz") as tf:
        try:
            # Python 3.12+ (and 3.10.12+/3.11.4+ backports): the 'data' filter
            # blocks path traversal, absolute paths, and links.
            tf.extractall(extract_dir, filter="data")
        except TypeError:
            # Older Pythons lack the extraction filter; validate members
            # ourselves before trusting the archive.
            dest = extract_dir.resolve()
            for member in tf.getmembers():
                target = (dest / member.name).resolve()
                if target != dest and dest not in target.parents:
                    raise ValueError(f"Unsafe path in archive: {member.name!r}")
                if member.issym() or member.islnk():
                    raise ValueError(f"Unsafe link in archive: {member.name!r}")
            tf.extractall(extract_dir)


# Stamped in the ``.complete`` marker when the remote hash could not be probed
# at extraction time. A later successful HEAD must not treat this as a hash
# mismatch, or a perfectly good local extraction would be re-downloaded.
_HASH_UNKNOWN = "unknown"


def _remote_archive_hash(url: str) -> str | None:
    """Best-effort content hash of the remote archive via a single HEAD request.

    Returns ``None`` when it cannot be determined (offline, ``HF_HUB_OFFLINE``,
    network error, rate limited) so callers can fall back to trusting an
    existing extraction rather than failing.
    """
    try:
        return get_hf_file_metadata(url).etag
    except Exception:
        return None


def ensure_archive(dl_manager, variant: str) -> pathlib.Path:
    """Ensure a variant's archive is extracted locally and return its directory.

    Downloads the single ``.tar.gz`` for ``variant`` (one request, so it avoids
    the per-file rate limits of fetching thousands of individual files),
    extracts it, and deletes the archive so only the extracted files remain on
    disk. This keeps steady-state disk usage at 1x instead of keeping both the
    archive and its unpacked copy.

    Idempotent and safe to call on every ``init()``: the ``.complete`` marker
    records the content hash of the archive it was extracted from. On each call
    a cheap HEAD fetches the current remote hash; extraction is (re)done only
    when the marker is missing/partial or the hash differs, so an unchanged
    archive is never re-downloaded (even across dataset revisions) while a
    genuine content change triggers a refresh. If the hash cannot be fetched
    (offline), an existing extraction is trusted. A file lock serializes
    concurrent callers (e.g. multi-worker data loaders).

    :param dl_manager: A datasets download manager (used to fetch the archive).
    :param variant:    One of ``"npz"``, ``"raw"``, ``"tetwild"``.
    :returns: Directory the archive was extracted into.
    """
    archive_name, verify_subdir = DatasetConfig.ARCHIVES[variant]
    download_config = getattr(dl_manager, "download_config", None)
    force_download = getattr(download_config, "force_download", False)
    extract_dir = _variant_extract_dir(download_config, variant)
    marker = extract_dir / ".complete"
    lock_path = _variant_lock_path(extract_dir)
    url = f"{DatasetConfig.REPO_URL}/{archive_name}"

    # Content hash of the remote archive (best-effort; None when unreachable).
    remote_hash = _remote_archive_hash(url)

    def _is_ready() -> bool:
        if force_download or not marker.is_file():
            return False
        if remote_hash is None:
            # Offline/unknown: trust the existing extraction rather than fail.
            return True
        try:
            stamped = marker.read_text().strip()
        except OSError:
            return False
        # Extraction stamped before its hash was known (transient HEAD failure):
        # trust it rather than forcing a re-download once HEAD recovers.
        if stamped == _HASH_UNKNOWN:
            return True
        return stamped == remote_hash

    # Fast path: extraction present and up to date -- avoid taking the lock.
    if _is_ready():
        return extract_dir

    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        # Re-check under the lock: another process may have just finished.
        if not _is_ready():
            # Clear any stale/partial/outdated extraction before re-extracting.
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            archive = dl_manager.download(url)
            extract_dir.mkdir(parents=True, exist_ok=True)
            _extract_archive(archive, extract_dir)
            # Reclaim the archive space -- the extracted files are what we keep.
            try:
                os.remove(archive)
            except OSError:
                logger.warning(
                    f"Could not remove archive after extraction: {archive}"
                )
            # Stamp the archive's content hash so a future change triggers a
            # refresh while an unchanged archive never re-downloads. Re-probe if
            # the earlier HEAD failed but the download itself succeeded; if still
            # unknown, stamp the sentinel so a later HEAD doesn't force a
            # needless re-download of a valid extraction.
            marker.write_text(
                remote_hash or _remote_archive_hash(url) or _HASH_UNKNOWN
            )

    verify_dir = extract_dir / verify_subdir
    if not verify_dir.is_dir():
        raise FileNotFoundError(
            f"Expected '{verify_subdir}' not found in extracted {variant} archive: "
            f"{extract_dir}"
        )
    return extract_dir


def clear_extracted(download_config, variant: str) -> bool:
    """Delete a variant's extracted files (and its lock) from the cache.

    Removes the ``thingi10k_<variant>_extracted`` directory that
    :func:`ensure_archive` unpacks the archive into, along with the sibling
    ``.lock`` file used to serialize concurrent extractions. A subsequent
    :func:`ensure_archive` call re-downloads and re-extracts as needed.

    :param download_config: A datasets download config (its ``cache_dir``
                            selects which cache location is cleared).
    :param variant:         One of ``"npz"``, ``"raw"``, ``"tetwild"``.
    :returns: ``True`` if an extracted directory was removed, ``False`` if none
        existed.
    """
    extract_dir = _variant_extract_dir(download_config, variant)
    lock_path = _variant_lock_path(extract_dir)

    removed = False
    # Serialize with any concurrent extraction before deleting the directory.
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
            removed = True
    # Leave the (empty) lock file in place: deleting it after releasing the lock
    # would let a concurrent ensure_archive create a fresh lock file on a new
    # inode, so two processes could believe they hold the extraction lock.
    return removed


class Thingi10KBuilder(datasets.GeneratorBasedBuilder):
    """
    Thingi10K Dataset builder.
    """

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="npz",
            version="1.2.0",
            description="Dataset stored in .npz format.",
        ),
        datasets.BuilderConfig(
            name="raw",
            version="1.1.0",
            description="Dataset stored in their original raw mesh format.",
        ),
        datasets.BuilderConfig(
            name="tetwild",
            version="1.1.0",
            description="Dataset remeshed using TetWild.",
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

        file_types = {
            "contextual_data": f"{metadata_url}/contextual_data.csv",
            "input_summary": f"{metadata_url}/input_summary.csv",
            "tag_data": f"{metadata_url}/tag_data.csv",
        }

        # Geometry data file depends on config
        match self.config.name:
            case "raw" | "npz":
                file_types["geometry_data"] = f"{metadata_url}/geometry_data.csv"
            case "tetwild":
                file_types["geometry_data"] = (
                    f"{metadata_url}/tetwild_geometry_data.csv"
                )
            case _:
                raise ValueError(f"Unknown config name: {self.config.name}")

        files = {}
        for file_type, url in file_types.items():
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
                df = pl.read_csv(file_path, schema_overrides=schema, ignore_errors=True)
                if "self_intersecting" not in df.columns:
                    df = df.with_columns(
                        (pl.col("num_self_intersections") > 0)
                        .cast(pl.Boolean)
                        .alias("self_intersecting")
                    )
                dataframes["geometry_data"] = df
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
        """Prepare the dataset files for download.

        Every variant downloads a single archive, extracts it, and deletes the
        archive so only the unpacked files stay on disk (1x, not 2x).
        """
        file_ids = geometry_data["file_id"]

        # Download + extract (archive deleted, hash-invalidated); paths are then
        # resolved relative to the returned extraction directory per variant.
        extraction_dir = ensure_archive(dl_manager, self.config.name)

        if self.config.name == "raw":
            # Raw meshes keep their original extension, derived per file from
            # the summary's Link column.
            raw_dir = extraction_dir / "Thingi10K" / "raw_meshes"
            downloaded_files = []
            for file_id, link in summary_data.select(["ID", "Link"]).iter_rows():
                if file_id in DatasetConfig.CORRUPT_FILE_IDS:
                    continue
                if link is None:
                    # No Link means no extension to resolve; skip rather than
                    # crash the whole raw-variant init on one bad row.
                    logger.warning(
                        f"Skipping raw file {file_id}: missing 'Link' column value."
                    )
                    continue
                ext = link.split(".")[-1].lower()
                downloaded_files.append(raw_dir / f"{file_id}.{ext}")
            return downloaded_files

        # npz and tetwild: one "<file_id>.npz" per file under a fixed subdir.
        npz_subdir = {"npz": "npz", "tetwild": "tetwild/10k_surface_npz"}.get(
            self.config.name
        )
        if npz_subdir is None:
            raise ValueError(f"Unknown config name: {self.config.name}")
        return [
            extraction_dir / npz_subdir / f"{file_id}.npz"
            for file_id in file_ids
            if file_id not in DatasetConfig.CORRUPT_FILE_IDS
        ]

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
                "self_intersecting": metadata["self_intersecting"],
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
