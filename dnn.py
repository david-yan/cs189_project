import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import combinations_with_replacement
from math import ceil, floor, sqrt

import utils
from polynomial_regression import PolynomialRegressor

def visualize_model(model, c_true, layer_width, num_layers):
    x = np.array([np.arange(-1 * x_range, x_range)])
    def input_x():
        return {"x": x.T}
    predictions = model.predict(input_fn=input_x)
    y = [next(predictions)["predictions"][0] for _ in range(len(x[0]))]
    terms = list(combinations_with_replacement(range(dim+1), degree))
    y_true = generate_Y(len(x.T), degree, dim, x.T, terms, c_true, False).T
    plt.title("Layer width: "+str(layer_width)+" Number of layers: "+str(num_layers))
    plt.plot(x[0], y, label='neural net')
    plt.plot(x[0], y_true[0], label='true model')
    plt.legend()
    plt.show()

def train_over_params(num_layers_list, layer_widths, x_train, y_train, c_true, x_test, y_test):

    x = np.array([np.linspace(-5, 5)])
    input_x = tf.estimator.inputs.numpy_input_fn({"x": x.T}, shuffle=False)
    terms = list(combinations_with_replacement(range(dim+1), degree))
    y_true = utils.generate_Y(len(x.T), degree, dim, x.T, terms, c_true, False).T

    input_train = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, shuffle=False, batch_size=x_train.shape[0], num_epochs=None)
    input_test = tf.estimator.inputs.numpy_input_fn({"x": x_test}, y_test, shuffle=False, batch_size=x_test.shape[0], num_epochs=None)

    feature_columns = [
        tf.feature_column.numeric_column(key="x")
    ]

    training_losses = []
    test_losses = []
    plt.figure(1, (20, 20))
    n_rows = len(num_layers_list)
    n_cols = len(layer_widths)
    for i in range(len(num_layers_list)):
        for j in range(len(layer_widths)):
            layer_width = layer_widths[j]
            num_layers = num_layers_list[i]
            model = tf.estimator.DNNRegressor(hidden_units=[layer_width]*num_layers, feature_columns=feature_columns)
            print("Training model")
            model.train(input_fn=input_train, steps=num_steps)
            print("Training done")
            train_result = model.evaluate(input_fn=input_train, steps=1)
            test_result = model.evaluate(input_fn=input_test, steps=1)
            training_losses.append(train_result["average_loss"])
            test_losses.append(test_result["average_loss"])
            predictions = model.predict(input_fn=input_x)
            y = [next(predictions)["predictions"][0] for _ in range(len(x[0]))]
            plt.subplot(n_rows, n_cols, (i*len(layer_widths)+j)+1)
            plt.title(
                "Layer width: "+str(layer_width)+" Number of layers: "+str(num_layers) \
                +"\nTraining: "+str(train_result["average_loss"])+" Test: " \
                +str(test_result["average_loss"]))
            plt.plot(x[0], y, label='neural net')
            plt.plot(x[0], y_true[0], label='true model')

    plt.tight_layout(pad=4.5)
    plt.show()

    plt.title("Losses")
    if len(num_layers_list) == 1:
        plt.plot(layer_widths, training_losses, label="training losses")
        plt.plot(layer_widths, test_losses, label="test losses")
        plt.legend()
        plt.show()
    if len(layer_widths) == 1:
        plt.plot(num_layers_list, training_losses, label="training losses")
        plt.plot(num_layers_list, test_losses, label="test losses")
        plt.legend()
        plt.show()

def evaluate_model_over_varying_degrees(model, num_steps, n_train, n_test, x_range, noise_Y=0, noise_X=0):
    train_errors = []
    test_errors = []

    plt.figure(1, (20, 20))
    x = np.linspace(-1*x_range/2, x_range/2)
    input_x = tf.estimator.inputs.numpy_input_fn({"x": np.array([x]).T}, shuffle=False)
    for degree in range(1, 7):
        c_true = utils.polynomial_coefficients(degree)
        x_train, y_train = utils.generate_data_from_c(n_train, degree, dim, x_range, c_true, noise_Y=noise_Y, noise_X=noise_X)
        x_test, y_test = utils.generate_data_from_c(n_test, degree, dim, x_range, c_true, noise_Y=noise_Y, noise_X=noise_X)
        input_train = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, shuffle=False, batch_size=x_train.shape[0], num_epochs=None)
        input_test = tf.estimator.inputs.numpy_input_fn({"x": x_test}, y_test, shuffle=False, batch_size=x_test.shape[0], num_epochs=None)
        model.train(input_fn=input_train, steps=num_steps)

        train_result = model.evaluate(input_fn=input_train, steps=1)
        test_result = model.evaluate(input_fn=input_test, steps=1)
        train_errors.append(train_result["average_loss"])
        test_errors.append(test_result["average_loss"])

        terms = list(combinations_with_replacement(range(2), degree))
        y_true = utils.generate_Y(len(x), degree, 1, np.array([x]).T, terms, c_true, 0)
        predictions = model.predict(input_fn=input_x)
        y_hat = [next(predictions)["predictions"][0] for _ in range(len(x))]

        plt.subplot(3, 2, degree)
        plt.title("Degree: "+str(degree))
        plt.plot(x, y_true, label='true model')
        plt.plot(x, y_hat, label='predicted model')
        plt.scatter(x_train, y_train, label='training data')
        plt.legend()
    plt.show()

    degrees = np.arange(1, 7)
    plt.title("MSE over degree")
    plt.plot(degrees, train_errors, label='training MSE')
    plt.plot(degrees, test_errors, label='test MSE')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    degree = 6
    dim = 1

    n_train = 100
    n_test = 20
    x_range = 7
    c_range = 20
    c_true = utils.polynomial_coefficients(degree)

    learning_rate = 0.1
    num_steps = 1000
    batch_size = 128
    display_step = 100

    num_input = 1
    num_output = 1

    feature_columns = [
        tf.feature_column.numeric_column(key="x")
    ]

    model = tf.estimator.DNNRegressor(hidden_units=[25]*12, feature_columns=feature_columns)
    evaluate_model_over_varying_degrees(model, num_steps, n_train, n_test, x_range, noise_Y=2, noise_X=0)


    # x_train, y_train = utils.generate_data_from_c(n_train, degree, dim, x_range, c_true, noise_Y=1)
    # x_test, y_test = utils.generate_data_from_c(n_test, degree, dim, x_range, c_true, noise_Y=1)
    #
    # plt.scatter(x_train.T[0], y_train.T[0])
    # plt.scatter(x_test.T[0], y_test.T[0])
    # plt.show()
    #
    # poly = PolynomialRegressor(dim=dim, degree=degree)

    # train_over_params([8], [10, 15, 20], x_train, y_train, c_true, x_test, y_test)
