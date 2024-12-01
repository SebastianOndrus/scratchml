import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from scratchml.models.kmeans import KMeans
from ..utils import repeat

class Test_KMeansWithMocks(unittest.TestCase):

    @patch('scratchml.models.kmeans.convert_array_numpy')
    @patch('scratchml.models.kmeans.euclidean')
    @repeat(1)
    def test_kmeans_mock(self, mock_euclidean, mock_convert_array_numpy):
        X = np.random.rand(10, 3)
        mock_euclidean.return_value = np.random.rand(10, 3)
        mock_convert_array_numpy.side_effect = lambda x: x
        
        kmeans = KMeans(n_init=1, n_clusters=3)
        kmeans.cluster_centers_ = np.random.rand(3, 3)
        
        predictions = kmeans.predict(X)
        
        # Verify the predictions
        mock_euclidean.assert_called()
        mock_convert_array_numpy.assert_called()
        self.assertEqual(predictions.shape, (10,))

if __name__ == "__main__":
    unittest.main()