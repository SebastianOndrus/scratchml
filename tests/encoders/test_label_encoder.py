from numpy.testing import assert_equal
from sklearn.preprocessing import LabelEncoder as SkLabelEncoder
from scratchml.encoders import LabelEncoder
import unittest
import numpy as np

class Test_LabelEncoder(unittest.TestCase):
    """
    Unittest class created to test the Label Encoder technique.
    """

    def test_1(self):
        """
        Test the Label Encoder implementation on a toy-problem and then compares
        it to the Scikit-Learn implementation.
        """
        le = LabelEncoder()
        le.fit([1, 2, 2, 6])
        le_transform = le.transform([1, 1, 2, 6])
        le_itransform = le.inverse_transform([0, 0, 1, 2])

        skle = SkLabelEncoder()
        skle.fit([1, 2, 2, 6])
        skle_transform = skle.transform([1, 1, 2, 6])
        skle_itransform = skle.inverse_transform([0, 0, 1, 2])

        assert_equal(le.classes_, skle.classes_)
        assert_equal(type(le.classes_), type(skle.classes_))
        assert_equal(le_transform, skle_transform)
        assert_equal(type(le_transform), type(skle_transform))
        assert_equal(le_itransform, skle_itransform)
        assert_equal(type(le_itransform), type(skle_itransform))

    def test_2(self):
        """
        Test the Label Encoder implementation on another toy-problem and then compares
        it to the Scikit-Learn implementation.
        """
        le = LabelEncoder()
        le.fit(["paris", "paris", "tokyo", "amsterdam"])
        le_transform = le.transform(["tokyo", "tokyo", "paris"])
        le_itransform = le.inverse_transform([2, 2, 1])

        skle = SkLabelEncoder()
        skle.fit(["paris", "paris", "tokyo", "amsterdam"])
        skle_transform = skle.transform(["tokyo", "tokyo", "paris"])
        skle_itransform = skle.inverse_transform([2, 2, 1])

        assert_equal(le.classes_, skle.classes_)
        assert_equal(type(le.classes_), type(skle.classes_))
        assert_equal(le_transform, skle_transform)
        assert_equal(type(le_transform), type(skle_transform))
        assert_equal(le_itransform, skle_itransform)
        assert_equal(type(le_itransform), type(skle_itransform))

        ## I have added the test_fit_transform method becouse 
    ## this method is not tested in the original test file
    ## ALso I have added tests for Error cases
    ## Whole class is now covered

    def test_3(self):
        print("Testing fit_transform method##############################################")
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

    def test_4(self):
        encoder = LabelEncoder()
        y = "not an array"
        
        with self.assertRaises(TypeError):
            encoder.fit(y)
    
    def test_5(self):
        encoder = LabelEncoder()
        encoder.fit(['a', 'b', 'c'])
        y = "not an array"
        
        with self.assertRaises(TypeError):
            encoder.transform(y)

    def test_6(self):
        encoder = LabelEncoder()
        y1 = np.array(['apple'])
        y2 = np.array(['banana'])
        
        with self.assertRaises(RuntimeError):
            encoder.fit(y1, y2)

    def test_7(self):
        encoder = LabelEncoder()
        encoder.fit(['a', 'b', 'c'])
        y1 = np.array(['a'])
        y2 = np.array(['b'])
        
        with self.assertRaises(RuntimeError):
            encoder.transform(y1, y2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
