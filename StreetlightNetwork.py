import numpy as np

# Predicts how safe it is to cross the road on a given color combination
def ele_mul(a, b):
    assert (len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output = output + (a[i] * b[i])
    return output



def neural_network(inputs, weights):
    pred = ele_mul(inputs,  weights)
    return pred


def predict_walk_or_stop(input_data, weights):
    prediction = neural_network(input_data, weights)
    print("Prediction result: " + str(prediction))
    if (prediction >= 0.5):
        return "Walk"
    else:
        return "Stop"



streetlights = np.array([[1, 0, 1],
          [0, 1, 1],
          [0, 0, 1],
          [1, 1, 1],
          [0, 1, 1],
          [1, 0, 1]])

walk_vs_stop = np.array([[0],
                        [1],
                        [0],
                        [1],
                        [1],
                        [0]])

alpha = 0.1
input = streetlights[0]
goal_prediction = walk_vs_stop[0]

weights = [0.2, 0.56, -0.8]


for i in range(1000):
    error_for_all_lights = 0
    for row_index in range(len(walk_vs_stop)):
        input = streetlights[row_index]
        goal_prediction = walk_vs_stop[row_index]

        pred = neural_network(input, weights)

        error = (pred - goal_prediction) ** 2
        error_for_all_lights = error_for_all_lights + error

        delta = pred - goal_prediction
        delta_weights = delta * input
        weights = weights - (delta_weights * alpha)
        print("Prediction: " + str(pred))
    print("Error: " + str(error_for_all_lights) + "\n" )
print("Weights: " + str(weights) + "\n")
print("--------------")

test_example = np.array([0, 1, 1])
test_prediction = predict_walk_or_stop(test_example, weights)
print("Result: " + test_prediction)