from scratchml.models.linear_regression import LinearRegression
from scratchml.utils import train_test_split
from sklearn.datasets import make_regression

if __name__ == "__main__":
    # generating a dataset for the regression task
    X, y = make_regression(
        n_samples=10000, n_features=5, n_targets=1, shuffle=True, noise=30
    )

    # splitting the data into training and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=True, stratify=False
    )

    # creating a linear regression model
    lr = LinearRegression(
        learning_rate=0.1,
        tol=1e-06,
        max_iters=-1,
        loss_function="mse",
        regularization=None,
        n_jobs=None,
    )

    # fitting the model
    lr.fit(X=X_train, y=y_train)

    # assessing the model's performance
    score = lr.score(X=X_test, y=y_test, metric="r_squared")

    print(score)
