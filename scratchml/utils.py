import numpy as np
from typing import Any, Union, Tuple, List

def KFold(
    X: np.ndarray,
    y: np.ndarray = None,
    stratify: bool = False,
    shuffle: bool = True,
    n_splits: int = 2
) -> List:
    """
    Splits the data into training and test set using the KFold technique.

    Args:
        X (np.ndarray): the feature array.
        y (np.ndarray, optional): the labels array. Will be ignored with
            stratify is False. Defaults to None.
        stratify (bool, optional): whether to stratify the set based on
            the labels (in other words, split the data while mantaining the classes
            distribution )or not. Defaults to False.
        shuffle (bool, optional): whether to shuffle the data or not. Defaults to True.
        n_splits (int, optional): the number of folds that the data will
            be split into. Defaults to 2.

    Returns:
        List: a list containing the training and test set for each fold.
    """
    X = convert_array_numpy(X)
    y = convert_array_numpy(y)

    # validating the number of splits
    try:
        assert n_splits >= 2
    except AssertionError:
        raise ValueError("N splits value should be equal or larget than 2.\n")
    
    # validating if the y is not None when stratify is True
    if stratify:
        try:
            assert y != None
        except AssertionError:
            raise ValueError("Y can not be None when stratify is True.\n")

    indices = np.arange(X.shape[0])

    # shuffling the array
    if shuffle:
        np.random.shuffle(indices)
    
    division_mod = X.shape[0] % n_splits
    fold_size = X.shape[0] // n_splits

    # amount of indexes per fold
    indexes = np.zeros(n_splits, dtype=int) + fold_size

    # creating a mapping array that will be used to add
    # extra samples to the folds
    # e.g.: number of samples = 101 and number of splits = 3
    # 101 % 3 = 2, so we have 2 left overs (which will be added
    # to the first two folds) => [34 34 33]
    extra_sample = np.zeros(n_splits, dtype=int)
    folds_indexes = []

    # if there are left over samples, we need to add thoses
    # indexes into the respective folds
    if division_mod != 0:
        extra_sample[:division_mod] = 1
        indexes = np.add(indexes, extra_sample)
    
    # splitting the indexes into folds
    if stratify:
        # analysing the classes distribution
        unique, counts = np.unique(y, return_counts=True)
        counts = np.asarray((unique, counts)).T
        classes_distribution = [
            (u, c)
            for u, c in counts
        ]

        # getting the indexes of each class considering how many
        # times each one occurred on the sample and splitting it
        # using the train test ratio
        # e.g.: [[0 100], [1 200]] => classes distribuition: 66%
        # class 1 and 33% class 0. If we want the train set to be
        # composed of 80% of the data, so we will have 158 (0.8 * 0.6 * 300)
        # samples for class 1 and 79 (0.8 * 0.33 * 300) for class 0
        for i in indexes:
            _temp_indexes = []

            for c, d in classes_distribution:
                _y = np.argwhere(y == c).reshape(-1)
                _size = int(d * i)
                _temp_indexes.extend(_y[:_size])
                indices = np.delete(indices, _y[:_size], axis=0)
            
            folds_indexes.append(_temp_indexes)
    else:
        for i in indexes:
            folds_indexes.append(indices[:i])
            indices = indices[i:]

    # organizing the folds into training and test
    test_fold = np.arange(n_splits)
    folds = []

    for tf in test_fold:
        # merging the indexes that are different from the
        # test folder index together to form the training indexes set
        training_indexes = [
            folds_indexes[i]
            for i in range(len(folds_indexes))
            if i != tf
        ]
        test_indexes = folds_indexes[tf]

        # converting the indexes lists to numpy array
        training_indexes = convert_array_numpy(training_indexes)
        training_indexes = training_indexes.reshape(-1)
        training_indexes = training_indexes.astype(int)

        test_indexes = convert_array_numpy(test_indexes)
        test_indexes = test_indexes.astype(int)
        
        folds.append((training_indexes, test_indexes))

    return folds

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: Union[float, int] = None,
    train_size: Union[float, int] = None,
    shuffle: bool = True,
    stratify: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into one training set and one validation set.

    Args:
        X (np.ndarray): the features array.
        y (np.ndarray): the labels array.
        test_size (Union[float, int], optional): the test set size
            (in total samples or the ratio of the entire dataset). Defaults to None.
        train_size (Union[float, int], optional): the train set size
            (in total samples or the ratio of the entire dataset). Defaults to None.
        shuffle (bool, optional): whether to shuffle the data or not. Defaults to True.
        stratify (bool, optional): whether to stratify the set based on
            the labels (in other words, split the data while mantaining the classes
            distribution )or not. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: the X train,
            X test, y train, and y test sets, respectively.
    """
    X = convert_array_numpy(X)
    y = convert_array_numpy(y)
    
    # validating the train_size and test_size parameters
    # as just one of them should be used
    try:
        assert not ((train_size == None) and (test_size == None)) or\
                ((train_size != None) and (test_size != None))
    except AssertionError:
        raise RuntimeError(
            f"You should pass train_size or test_size, not both or neither.\n"
        )

    # validating the test size parameter
    if test_size != None:
        if isinstance(test_size, float):
            try:
                assert 0 < test_size < 1
                test_split_ratio = test_size
                
            except AssertionError:
                raise ValueError("Test size value should be between 0 and 1.\n")
        elif isinstance(test_size, int):
            try:
                assert 0 < test_size < X.shape[0]
                test_split_ratio = test_size / X.shape[0]
            except AssertionError:
                raise ValueError(
                    f"Test size value should be between 0 and {X.shape[0]}.\n"
                )
    
    # validating the train size parameter
    if train_size != None:
        if isinstance(train_size, float):
            try:
                assert 0 < train_size < 1
                train_split_ratio = train_size
            except AssertionError:
                raise ValueError("Train size value should be between 0 and 1.\n")
        elif isinstance(train_size, int):
            try:
                assert 0 < train_size < X.shape[0]
                train_split_ratio = train_size / X.shape[0]
            except AssertionError:
                raise ValueError(
                    f"Train size value should be between 0 and {X.shape[0]}.\n"
                )
    
    # defining the split ratio of the train set
    if train_size == None:
        train_split_ratio = 1 - test_split_ratio

    # shuffling the arrays
    if shuffle:
        shuffled_indices = np.arange(X.shape[0])
        np.random.shuffle(shuffled_indices)

        X = X[shuffled_indices]
        y = y[shuffled_indices]
    
    if stratify:
        # analysing the classes distribution
        unique, counts = np.unique(y, return_counts=True)
        counts = np.asarray((unique, counts)).T
        classes_distribution = [
            (u, c)
            for u, c in counts
        ]

        train_indexes = []
        test_indexes = []

        # getting the indexes of each class considering how many
        # times each one occurred on the sample and splitting it
        # using the train test ratio
        # e.g.: [[0 100], [1 200]] => classes distribuition: 66%
        # class 1 and 33% class 0. If we want the train set to be
        # composed of 80% of the data, so we will have 158 (0.8 * 0.6 * 300)
        # samples for class 1 and 79 (0.8 * 0.33 * 300) for class 0
        for c, d in classes_distribution:
            _y = np.argwhere(y == c).reshape(-1)
            _size = int(d * train_split_ratio)

            train_indexes.extend(_y[:_size])
            test_indexes.extend(_y[_size:])

        X_train = X[train_indexes]
        X_test = X[test_indexes]

        y_train = y[train_indexes]
        y_test = y[test_indexes]

    else:
        # splitting the arrays sequentially
        train_indexes = int(train_split_ratio * X.shape[0])

        X_train = X[:train_indexes]
        X_test = X[train_indexes:]

        y_train = y[:train_indexes]
        y_test = y[train_indexes:]

    # converting the train and test sets to numpy arrays
    X_train = convert_array_numpy(X_train)
    X_test = convert_array_numpy(X_test)
    y_train = convert_array_numpy(y_train)
    y_test = convert_array_numpy(y_test)
    
    return X_train, X_test, y_train, y_test

def convert_array_numpy(
    array: Any
) -> np.ndarray:
    """
    Auxiliary function that converts an array to numpy array.

    Args:
        array (Any): the array that will be converted.

    Returns:
        array (np.ndarray): the converted numpy array.
    """
    if isinstance(array, list):
        array = np.asarray(array, dtype="O")
        return array
    if isinstance(array, np.ndarray):
        return array
    else:
        raise TypeError("Invalid type. Should be np.ndarray or list.\n")