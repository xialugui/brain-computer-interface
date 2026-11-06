from ast import pattern
import copy
import array
import numpy as np


class HopfieldNetwork:
    def __init__(self, number_of_neuron: int):
        self.number_of_neuron = number_of_neuron
        self.weights = np.zeros((number_of_neuron, number_of_neuron))

    def fit(self, patterns):
        for i in range(self.number_of_neuron):
            for j in range(self.number_of_neuron):
                if i == j:
                    self.weights[i, j] = 0
                else:
                    self.weights[i, j] = np.sum(
                        [patterns[k][i] * patterns[k][j] for k in range(len(patterns))]
                    )
        print(self.weights, "----------\n")
        # self.weights = patterns.T @ patterns  # 形状 (N, N)
        # np.fill_diagonal(self.weights, 0)

    def recall(self, input):
        inputs = copy.deepcopy(input)
        new_inputs = np.zeros(self.number_of_neuron)
        while np.sum(np.abs(inputs - new_inputs)) != 0:
            new_inputs = copy.deepcopy(x=inputs)
            for i in range(self.number_of_neuron):
                temp = np.dot(inputs, self.weights[:, i])
                if temp > 0:
                    inputs[i] = 1
                else:
                    inputs[i] = -1
        return inputs
