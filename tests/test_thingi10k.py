import pytest
import thingi10k
from pathlib import Path

def test_thingi10k():
    """ Simple test to ensure everything is working """
    thingi10k.init()
    high_genus_dataset = thingi10k.dataset(genus=(1000, None))
    assert len(high_genus_dataset) == 3

    for entry in high_genus_dataset:
        assert Path(entry['file_path']).exists()
        V, F = thingi10k.load_file(entry['file_path'])
        assert V.shape[0] == entry['num_vertices']
        assert F.shape[0] == entry['num_facets']


def test_clear_cache(tmp_path):
    """clear_cache removes the extracted folder (and reports when absent)."""
    cache_dir = str(tmp_path)

    # Nothing extracted yet -> reports False via logging, removes nothing.
    thingi10k.clear_cache(variant="npz", cache_dir=cache_dir)

    # Simulate an extracted variant folder, then clear it.
    extracted = tmp_path / "thingi10k_npz_extracted"
    (extracted / "npz").mkdir(parents=True)
    (extracted / ".complete").write_text("deadbeef")
    assert extracted.exists()

    thingi10k.clear_cache(variant="npz", cache_dir=cache_dir)
    assert not extracted.exists()


def test_clear_cache_rejects_bad_variant():
    with pytest.raises(ValueError):
        thingi10k.clear_cache(variant="bogus")
