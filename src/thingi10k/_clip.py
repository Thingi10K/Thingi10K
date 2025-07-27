"""
CLIP-based feature extraction and similarity search for the Thingi10K dataset.

This module provides functionality to use CLIP (Contrastive Language-Image Pre-training)
models for searching 3D models in the Thingi10K dataset based on text queries.
It handles downloading pre-computed CLIP features and provides an interface for
semantic search of 3D models.

.. note::
    This module requires additional dependencies (torch, open_clip) that can be
    installed with: ``pip install thingi10k[clip]``
"""

try:
    import torch
    import open_clip # type: ignore[import-untyped]
    from ._builder import DatasetConfig

    with_clip = True
except ImportError as e:
    with_clip = False

import datasets # type: ignore[import-untyped]
import pathlib
from ._logging import logger


class ClipFeatures:
    """
    CLIP-based feature extractor and similarity search for 3D models.

    This class provides functionality to search for 3D models in the Thingi10K dataset
    using natural language queries. It downloads pre-computed CLIP features and uses
    a CLIP model to encode text queries for similarity matching.

    :param model_name: Name of the CLIP model to use for text encoding
    :type model_name: str

    .. note::
        Requires torch and open_clip to be installed. Install with:
        ``pip install thingi10k[clip]``

    Example:
        >>> clip_features = ClipFeatures("ViT-B-32-quickgelu")
        >>> results = clip_features.query("car wheel", threshold=0.1)
        >>> print(f"Found {len(results)} matching models")
    """

    def __init__(self, model_name="ViT-B-32-quickgelu", force_redownload=False):
        """
        Initialize the CLIP features extractor.

        :param model_name: Name of the CLIP model to use for encoding text queries.
                          Must match one of the available pre-computed feature sets.
        :type model_name: str
        :param force_redownload: Whether to force redownload of features even if they exist locally.
        :type force_redownload: bool
        :raises AssertionError: If CLIP dependencies are not available
        """
        if not with_clip:
            return

        self.model_name = model_name
        self.__initialize_features(force_redownload)
        self.__initialize_model()
        logger.info("CLIP features initialized successfully.")

    def __initialize_model(self):
        """
        Initialize the CLIP model and tokenizer.

        Downloads and loads the specified CLIP model with pre-trained weights
        from LAION-400M dataset. Also initializes the corresponding tokenizer.

        :raises Exception: If model initialization fails
        """
        logger.info(f"Initializing CLIP model: {self.model_name}")
        self.model, _, _ = open_clip.create_model_and_transforms(
            self.model_name, pretrained="laion400m_e32"
        )

        self.tokenizer = open_clip.get_tokenizer(self.model_name)

    def __initialize_features(self, force_redownload: bool):
        """
        Download and load pre-computed CLIP features for the Thingi10K dataset.

        Downloads the pre-computed image features for all models in the dataset,
        loads them into memory, and normalizes them for similarity computation.

        :param force_redownload: Whether to force redownload of features even if they exist locally.
        :type force_redownload: bool

        Also loads the corresponding file IDs for mapping results back to models.

        :raises Exception: If feature download or loading fails
        """
        logger.info("Downloading CLIP features for Thingi10K dataset.")
        REPO_URL = DatasetConfig.REPO_URL

        # Configure download manager with force redownload option
        download_config = None
        if force_redownload:
            download_config = datasets.DownloadConfig(force_download=True)

        dl_manager = datasets.DownloadManager(download_config=download_config)
        url = f"{REPO_URL}/clip_features/{self.model_name}.tar.gz"
        self.features_path = pathlib.Path(dl_manager.download_and_extract(url))

        features = []
        for i in range(1, 10):
            feature_file = self.features_path / f"image_features_{i}.pt"
            features.append(torch.load(feature_file))
        self.features = torch.cat(features, dim=0)
        self.file_ids = torch.load(self.features_path / "file_ids.pt")

        with torch.no_grad():
            self.features /= self.features.norm(dim=-1, keepdim=True)

    def query(self, query_text: str, threshold: float = 0.01):
        """
        Search for models in the dataset that match the query text using CLIP model.

        Encodes the input text query using the CLIP model and computes similarity
        scores with all pre-computed image features in the dataset. Returns file IDs
        of models that exceed the specified similarity threshold.

        :param query_text: The text query to search for in the dataset
        :type query_text: str
        :param threshold: The cutoff threshold within [0, 1] for matches.
                          0 will match everything, 1 matches nothing.
        :type threshold: float
        :return: Tensor containing file IDs of models that match the query text
        :rtype: torch.Tensor
        :raises AssertionError: If CLIP model is not available

        Example:
            >>> clip_features = ClipFeatures()
            >>> results = clip_features.query("A perfect sphere", threshold=0.1)
            >>> print(f"Found {len(results)} models matching the query")
        """
        assert (
            with_clip
        ), "CLIP model is not available. Please `pip install thingi10k[clip]`."
        text = self.tokenizer([query_text])

        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (
                (100.0 * self.features @ text_features.T).softmax(dim=0).squeeze()
            )
            selection = self.file_ids[text_probs > threshold]

            return selection
