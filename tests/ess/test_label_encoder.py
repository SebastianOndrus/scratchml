import unittest
import numpy as np
from scratchml.encoders import LabelEncoder
from ..utils import repeat

class Test_LabelEncoder(unittest.TestCase):
    ## I have added the test_fit_transform method becouse 
    ## this method is not tested in the original test file
    ## ALso I have added tests for Error cases
    ## Whole class is now covered

    @repeat(1)
    def test_fit_transform(self):
        encoder = LabelEncoder()
        y = np.array(['dog', 'cat', 'dog', 'mouse'])
        
        # Fit and transform
        transformed = encoder.fit_transform(y)
        
        # Check if the classes are correctly identified
        np.testing.assert_array_equal(encoder.classes_, np.array(['cat', 'dog', 'mouse']))
        
        # Check the transformation output
        expected_transformed = np.array([1, 0, 1, 2])
        np.testing.assert_array_equal(transformed, expected_transformed)
        
        # Check the internal mapping
        expected_map = {'cat': 0, 'dog': 1, 'mouse': 2}
        self.assertEqual(encoder.classes_map_, expected_map)

    @repeat(1)
    def test_fit_type_error(self):
        encoder = LabelEncoder()
        y = "not an array"
        
        with self.assertRaises(TypeError):
            encoder.fit(y)
    
    @repeat(1)
    def test_transform_type_error(self):
        encoder = LabelEncoder()
        encoder.fit(['a', 'b', 'c'])
        y = "not an array"
        
        with self.assertRaises(TypeError):
            encoder.transform(y)

    @repeat(1)
    def test_fit_runtime_error(self):
        encoder = LabelEncoder()
        y1 = np.array(['apple'])
        y2 = np.array(['banana'])
        
        with self.assertRaises(RuntimeError):
            encoder.fit(y1, y2)

    @repeat(1)
    def test_transform_runtime_error(self):
        encoder = LabelEncoder()
        encoder.fit(['a', 'b', 'c'])
        y1 = np.array(['a'])
        y2 = np.array(['b'])
        
        with self.assertRaises(RuntimeError):
            encoder.transform(y1, y2)

if __name__ == "__main__":
    unittest.main()