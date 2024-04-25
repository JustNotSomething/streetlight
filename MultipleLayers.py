import numpy as np
np.random.seed(1)


def relu(x):
    return (x > 0) * x


def relu2deriv(output):
    return output > 0


streetlights = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
])

walk_vs_stop = np.array([[1, 1, 0, 0]]).T

alpha = 0.2
hidden_size = 4

weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

for iteration in range(60):
    layer_2_error = 0
    for i in range(len(streetlights)):
        layer_0 = streetlights[i:i + 1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        layer_2_error = layer_2_error + np.sum((layer_2 - walk_vs_stop[i : i + 1]) ** 2)

        layer_2_delta = (layer_2 - walk_vs_stop[i:i + 1])
        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * relu2deriv(layer_1)

        weights_0_1 = weights_0_1 - alpha * np.dot(layer_0.T, layer_1_delta)
        weights_1_2 = weights_1_2 - alpha * np.dot(layer_1.T, layer_2_delta)

    if iteration % 10 == 9:
        print("Error: " + str(layer_2_error))

test_example = np.array([1, 1, 1])
test_example = test_example.reshape(1, -1)

test_layer_0 = relu(np.dot(test_example, weights_0_1))
test_layer_1 = np.dot(test_layer_0, weights_1_2)

if test_layer_1 >= 0.5:
    print("Walk")
else:
    print("Stop")
