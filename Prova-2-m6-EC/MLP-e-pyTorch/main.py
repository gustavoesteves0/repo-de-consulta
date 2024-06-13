import numpy as np
import math

class MLP:
    def __init__(self, input_size=2, hidden_size=2, output_size=1, learning_rate=0.1):
        self.learning_rate = learning_rate
        
        # Inicialização dos pesos e bias
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(output_size)
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _heaviside(self, x):
        return 1 if x >= 0.5 else 0

    def _mse(self, target, output):
        return np.mean((target - output) ** 2)
    
    def forward_pass(self, data):
        # Passagem direta
        self.hidden_input = np.dot(data, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self._sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self._sigmoid(self.final_input)
        
        return self.final_output

    def backward_pass(self, data, target):
        # Erro na camada de saída
        output_error = target - self.final_output
        output_delta = output_error * self._sigmoid_derivative(self.final_output)
        
        # Erro na camada escondida
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self._sigmoid_derivative(self.hidden_output)
        
        # Atualização dos pesos e bias
        self.weights_hidden_output += np.outer(self.hidden_output, output_delta) * self.learning_rate
        self.bias_output += output_delta * self.learning_rate
        
        self.weights_input_hidden += np.outer(data, hidden_delta) * self.learning_rate
        self.bias_hidden += hidden_delta * self.learning_rate

    def train(self, data, targets, epochs=10000):
        for epoch in range(epochs):
            for x, y in zip(data, targets):
                self.forward_pass(x)
                self.backward_pass(x, y)
                
    def predict(self, data):
        output = self.forward_pass(data)
        return [self._heaviside(x) for x in output]

# Dados de treino para a porta XOR
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Treinando o MLP
mlp = MLP()
mlp.train(data, targets)

# Testando o MLP
for x in data:
    print(f'Input: {x} -> Output: {mlp.predict(x)}')
