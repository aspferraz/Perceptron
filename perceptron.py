from random import uniform


class Perceptron:

    def __init__(self, input_size):
        self.inputLayerSize = input_size
        self.weights = []
        self.bias = uniform(0, 1)

        for i in range(input_size):
            self.weights.append(uniform(0, 1))

    @staticmethod
    def normalize_value(val):
        if val < 0:
            return -1
        else:
            return 1

    # nnInput is an array containing all input values
    def process_input(self, nn_input):
        # The number of elements in nnInput should be equal to the number of weights
        assert len(nn_input) == len(self.weights)

        # Initialize the weighted sum with the bias
        unprocessed_output_val = self.bias

        # Add the rest of the weighted sum
        for i in range(len(nn_input)):
            unprocessed_output_val += nn_input[i] * self.weights[i]

        return self.normalize_value(unprocessed_output_val)  # Return the formatted value

    def fit(self, training_set, max_iter, learning_rate):
        initial_weights = self.weights[:]
        print(initial_weights)
        print(self.bias)
        for epoch in range(max_iter):
            initial_weights = self.weights[:]
            for i in range(len(training_set)):
                self.train_on_input(training_set[i][0:self.inputLayerSize],
                                    training_set[i][self.inputLayerSize], learning_rate)

            if initial_weights == self.weights:
                print('epoch ', epoch)
                print(self.weights)
                print(self.bias)
                break

    def train_on_input(self, input_values, expected_output_val, learning_rate):
        nn_val = self.process_input(input_values)
        error = expected_output_val - nn_val
        self.adjust_for_error(input_values, error, learning_rate)

    def adjust_for_error(self, input_values, error, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] += error * input_values[i] * learning_rate
        self.bias += error * learning_rate
