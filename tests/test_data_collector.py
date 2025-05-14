import unittest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
import sys

# Add src directory to sys.path to import the collector
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_ingestion.barcelona_data_collector import BarcelonaDataCollector

class TestBarcelonaDataCollector(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.test_base_dir = Path("./test_data_dir")
        self.test_base_dir.mkdir(exist_ok=True)
        # Mock os.getenv to provide default API keys during tests
        self.env_patcher = patch.dict(os.environ, {
            'WEATHER_API_KEY': 'test_weather_key',
            'TMB_APP_ID': 'test_tmb_id',
            'TMB_APP_KEY': 'test_tmb_key'
        })
        self.env_patcher.start()
        self.collector = BarcelonaDataCollector(base_dir=str(self.test_base_dir))

    def tearDown(self):
        """Clean up test environment."""
        self.env_patcher.stop()
        # Clean up created directories and files (implement properly)
        # For now, just remove the base test dir if empty or handle cleanup
        if self.test_base_dir.exists():
             # Be careful with recursive deletion in real tests
             # shutil.rmtree(self.test_base_dir) 
             pass # Basic cleanup for now

    def test_initialization(self):
        """Test if the collector initializes correctly."""
        self.assertIsNotNone(self.collector)
        self.assertEqual(self.collector.base_dir, self.test_base_dir)
        self.assertTrue((self.test_base_dir / 'parking').exists())
        self.assertTrue((self.test_base_dir / 'weather').exists())
        # Check if config loads defaults (or mocked env vars)
        self.assertEqual(self.collector.config['weather_api_key'], 'test_weather_key')
        self.assertEqual(self.collector.config['tmb_app_id'], 'test_tmb_id')

    @patch('requests.get')
    def test_download_file_success(self, mock_get):
        """Test successful file download."""
        # Mock requests.get response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b"file content"
        mock_get.return_value = mock_response

        # Test download
        filepath = self.collector.download_file("http://example.com/data.txt", "data.txt", 'parking')
        
        self.assertIsNotNone(filepath)
        self.assertTrue(filepath.exists())
        self.assertEqual(filepath.name, "data.txt")
        self.assertEqual(filepath.parent.name, 'parking')
        
        # Verify requests.get was called
        mock_get.assert_called_once_with("http://example.com/data.txt", headers=None, timeout=30)
        
        # Clean up downloaded file
        if filepath and filepath.exists():
            filepath.unlink()

    @patch('requests.get')
    def test_download_file_failure(self, mock_get):
        """Test failed file download after retries."""
        # Mock requests.get to raise an exception
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        filepath = self.collector.download_file("http://example.com/bad_url.txt", "bad_url.txt", 'parking')
        
        self.assertIsNone(filepath)
        # Check if requests.get was called multiple times (retries)
        self.assertEqual(mock_get.call_count, self.collector.DOWNLOAD_RETRIES) # Assuming DOWNLOAD_RETRIES is accessible or hardcode 3

    # Add more tests for:
    # - Fallback mechanism in download_file
    # - ZIP file extraction
    # - _validate_csv method
    # - get_parking_data, get_weather_data, etc. (mocking API calls)
    # - _clean_data method
    # - integrate_data method (using dummy CSV files)

if __name__ == '__main__':
    unittest.main() 