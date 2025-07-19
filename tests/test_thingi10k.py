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
