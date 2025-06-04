import unittest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
import sys
import shutil
import tempfile
import requests
import json
import pandas as pd

# Add src directory to sys.path to import the collector
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_ingestion.barcelona_data_collector import BarcelonaDataCollector

class TestBarcelonaDataCollector(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        # Use tempfile for test directory
        self.test_dir = tempfile.mkdtemp()
        self.test_base_dir = Path(self.test_dir)
        
        # Create a mock pass.json file
        self.pass_json_path = Path("pass.json")
        self.pass_json_content = {
            "local": {
                "app_id": "test_tmb_id",
                "app_key": "test_tmb_key"
            }
        }
        with open(self.pass_json_path, 'w') as f:
            json.dump(self.pass_json_content, f)
        
        # Patch environment variables for all credentials
        self.env_patcher = patch.dict(os.environ, {
            'WEATHER_API_KEY': 'test_weather_key',
            'BCN_OD_TOKEN': 'test_bcn_token',
            'TMB_APP_ID': 'test_tmb_id',
            'TMB_APP_KEY': 'test_tmb_key',
            'METEO_CAT_API_KEY': 'test_weather_key',
        })
        self.env_patcher.start()
        self.collector = BarcelonaDataCollector(base_dir=str(self.test_base_dir))

    def tearDown(self):
        """Clean up test environment."""
        self.env_patcher.stop()
        if self.pass_json_path.exists():
            self.pass_json_path.unlink()
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test collector initialization."""
        self.assertEqual(self.collector.tmb_app_id, 'test_tmb_id')
        self.assertEqual(self.collector.tmb_app_key, 'test_tmb_key')
        self.assertEqual(self.collector.meteo_cat_api_key, 'test_weather_key')

    def test_download_file_success(self):
        """Test successful file download."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b'test content']
        
        with patch('requests.get', return_value=mock_response):
            filepath = self.collector.download_file(
                'http://test.com/file.txt',
                'test.txt',
                'test_folder'
            )
            self.assertIsNotNone(filepath)
            self.assertTrue(filepath.exists())
            self.assertEqual(filepath.read_text(), 'test content')

    def test_download_file_failure(self):
        """Test file download failure."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException()
        
        with patch('requests.get', return_value=mock_response):
            filepath = self.collector.download_file(
                'http://test.com/file.txt',
                'test.txt',
                'test_folder'
            )
            self.assertIsNone(filepath)

    def test_get_parking_data(self):
        """Test parking data collection."""
        # Mock successful response for BSM underground parking data
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [json.dumps({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": "Test Parking",
                        "capacity": 100
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [2.1734, 41.3851]
                    }
                }
            ]
        }).encode('utf-8')]
        
        with patch('requests.get', return_value=mock_response):
            result = self.collector.get_parking_data()
            self.assertIsInstance(result, dict)
            self.assertIn('Aparcaments_securitzat.json', result)
            # Check file content is valid JSON
            with open(result['Aparcaments_securitzat.json'], 'r') as f:
                data = json.load(f)
                self.assertEqual(data['type'], 'FeatureCollection')

    def test_get_weather_data(self):
        """Test weather data collection."""
        # Mock successful response for historical weather data
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b'test weather data']
        
        with patch('requests.get', return_value=mock_response):
            result = self.collector.get_weather_data()
            self.assertIsInstance(result, dict)
            self.assertIn('historical_weather.csv', result)

    def test_get_transport_data(self):
        """Test transport data collection."""
        # Mock successful response for GTFS data
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b"mock gtfs zip content"
        # For iBus, patch .json() to return a real dict
        def side_effect(*args, **kwargs):
            if 'ibus' in args[0]:
                return MagicMock(json=lambda: {'data': 'ibus'})
            return mock_response
        with patch('requests.get', return_value=mock_response) as mock_get:
            # Patch the .json method for the iBus call
            mock_response.json.return_value = {'data': 'ibus'}
            # Patch zipfile.ZipFile to simulate successful extraction
            with patch('zipfile.ZipFile', autospec=True) as mock_zip:
                mock_zip.return_value.__enter__.return_value.extractall.return_value = None
                result = self.collector.get_transport_data()
                self.assertIsInstance(result, dict)
                self.assertIn('tmb_gtfs', result)
                self.assertIn('tmb_ibus_stop_2775', result)
                # Check that the iBus file is valid JSON
                with open(result['tmb_ibus_stop_2775'], 'r') as f:
                    data = json.load(f)
                    self.assertEqual(data['data'], 'ibus')

    def test_get_events_data(self):
        """Test events data collection."""
        # Mock successful response for events data
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'success': True,
            'result': {
                'records': [
                    {'id': 1, 'name': 'Test Event'}
                ],
                'fields': [
                    {'id': 'id'},
                    {'id': 'name'}
                ]
            }
        }
        
        with patch('requests.get', return_value=mock_response):
            result = self.collector.get_events_data()
            self.assertIsInstance(result, dict)
            self.assertIn('cultural_events.csv', result)

    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Create test data with duplicates and missing values
        test_data = {
            'ID_TARIFA': [1, 1, 2, 2, 3],
            'value': [10, 10, 20, None, 30]
        }
        df = pd.DataFrame(test_data)
    
        # Clean the data
        cleaned_df = self.collector._clean_data(df)
    
        # Reset index before comparison
        cleaned_df = cleaned_df.reset_index(drop=True)
    
        # Verify cleaning results (ignore index)
        self.assertEqual(len(cleaned_df), 4)  # Only exact duplicates are removed
        expected_id_series = pd.Series([1, 2, 2, 3], name='ID_TARIFA')
        expected_value_series = pd.Series([10, 20, None, 30], name='value')
        pd.testing.assert_series_equal(cleaned_df['ID_TARIFA'], expected_id_series, check_dtype=False)
        pd.testing.assert_series_equal(cleaned_df['value'], expected_value_series, check_dtype=False)

if __name__ == '__main__':
    unittest.main() 