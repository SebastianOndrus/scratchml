import unittest
import numpy as np
from sklearn.metrics import mean_absolute_error
from scratchml.criterions import poisson, absolute_error

class Test_Criterions(unittest.TestCase):

    def test_poisson(self):
        print("Testing poisson")
        y = np.array([1, 2, 3, 4, 5])
        y_mean = np.mean(y)
        result_scratchml = poisson(y_mean, y)
        
        # Scikit-learn doesn't have a Poisson function
        # that has the same functionality as the one in scratchml
        # so we'll use the manual calculation
        manual_result = 0.71812556750
        self.assertAlmostEqual(result_scratchml, manual_result, places=5)

    def test_absolute_error(self):
        print("Testing absolute_error")
        y = np.array([1, 2, 3, 4, 5])
        y_median = np.median(y)
        result_scratchml = absolute_error(y_median, y)
        
        result_sklearn = mean_absolute_error(y, np.full_like(y, y_median))
        
        self.assertAlmostEqual(result_scratchml, result_sklearn, places=5)

if __name__ == "__main__":
    unittest.main()