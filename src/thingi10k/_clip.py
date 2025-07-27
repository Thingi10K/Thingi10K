try:
    import torch
    import open_clip
    from ._builder import DatasetConfig

    with_clip = True
except ImportError as e:
    with_clip = False

import datasets
import pathlib
from ._logging import logger


class ClipFeatures:
    def __init__(self, model_name="ViT-B-32-quickgelu"):
        if not with_clip:
            return

        self.model_name = model_name
        self.__initialize_features()
        self.__initialize_model()
        logger.info("CLIP features initialized successfully.")

    def __initialize_model(self):
        logger.info(f"Initializing CLIP model: {self.model_name}")
        self.model, _, _ = open_clip.create_model_and_transforms(
            self.model_name, pretrained="laion400m_e32"
        )

        self.tokenizer = open_clip.get_tokenizer(self.model_name)

    def __initialize_features(self):
        logger.info("Downloading CLIP features for ThingI10K dataset.")
        REPO_URL = DatasetConfig.REPO_URL

        dl_manager = datasets.DownloadManager()
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
        """Search for models in the dataset that match the query text using CLIP model.

        @param query_text: The text query to search for in the dataset.
        @param threshold:  The cutoff threshold within [0, 1] for matches. 0 will match everthing,
                           and 1 matches nothing. Default is 0.01.

        @return:          The file IDs of the models that match the query text.
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
