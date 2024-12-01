from scratchml.models.multilayer_perceptron import MLPClassifier, MLPRegressor
from scratchml.utils import KFold
from sklearn.datasets import make_classification, make_regression


def example_mlp_classifier() -> None:
    """
    Practical example of how to use the Multilayer Perceptron (MLP) Classifier model.
    """
    # generating a dataset for the classfication set
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_classes=2,
        n_clusters_per_class=1,
        n_informative=2,
        n_redundant=1,
        n_repeated=0,
        shuffle=True,
    )

    # splitting the data into training and testing using KFold
    folds = KFold(X, y, stratify=True, shuffle=True, n_splits=5)

    for fold, (train_indexes, test_indexes) in enumerate(folds):
        # getting the training and test sets
        X_train = X[train_indexes]
        y_train = y[train_indexes]

        X_test = X[test_indexes]
        y_test = y[test_indexes]

        # creating a MLP model instance
        mlp = MLPClassifier(
            loss_function="cross_entropy",
            hidden_layer_sizes=(
                32,
                64,
            ),
            max_iter=100,
            batch_size=64,
            verbose=0,
        )

        # fitting the model
        mlp.fit(X=X_train, y=y_train)

        # assessing the model's performance
        score = mlp.score(X=X_test, y=y_test, metric="accuracy")

        print(f"The model achieved an accuracy score of {score} on the fold {fold}.\n")


def example_mlp_regressor() -> None:
    """
    Practical example of how to use the Multilayer Perceptron (MLP) Regressor model.
    """
    # generating a dataset for the regression task
    X, y = make_regression(
        n_samples=2000, n_features=4, n_targets=1, shuffle=True, noise=0.1
    )

    # splitting the data into training and testing using KFold
    folds = KFold(X, y, stratify=False, shuffle=True, n_splits=5)

    for fold, (train_indexes, test_indexes) in enumerate(folds):
        # getting the training and test sets
        X_train = X[train_indexes]
        y_train = y[train_indexes]

        X_test = X[test_indexes]
        y_test = y[test_indexes]

        # creating an MLP Regressor instance
        mlp = MLPRegressor(
            loss_function="mse",
            hidden_layer_sizes=(32, 64),
            max_iter=100,
            batch_size=32,
            verbose=0,
            learning_rate_init=0.001,
        )

        # fitting the model
        mlp.fit(X=X_train, y=y_train)

        # assessing the model's performance
        score = mlp.score(X=X_test, y=y_test, metric="r_squared")

        print(f"The model achieved an R² score of {score} on the fold {fold}.\n")


if __name__ == "__main__":
    # example_mlp_classifier()
    example_mlp_regressor()
