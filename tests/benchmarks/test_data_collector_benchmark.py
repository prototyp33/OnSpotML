import pytest
from pathlib import Path
import tempfile
import os
from data_ingestion.barcelona_data_collector import BarcelonaDataCollector

@pytest.fixture
def collector():
    """Create a test collector instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = BarcelonaDataCollector(base_dir=tmpdir)
        yield collector

def test_download_file_benchmark(benchmark, collector):
    """Benchmark the download_file method."""
    # Mock the download_file method to avoid actual network calls
    def mock_download(url, filename, subdir):
        return Path(collector.base_dir) / subdir / filename
    
    collector.download_file = mock_download
    
    # Run the benchmark
    result = benchmark(
        collector.download_file,
        "http://example.com/test.txt",
        "test.txt",
        "parking"
    )
    
    assert result is not None
    assert result.name == "test.txt"
    assert result.parent.name == "parking" 