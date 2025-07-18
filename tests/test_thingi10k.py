import pytest
import thingi10k

def test_thingi10k():
    """ Simple test to ensure everything is working """
    thingi10k.init()
    high_genus_files = [entry['file_id'] for entry in thingi10k.dataset(genus=(1000, None))]
    assert len(high_genus_files) == 3
