
import math


def dot(vec1,vec2):
    return sum(x*y for x,y in zip(vec1,vec2))
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    outputs = []
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)
       

        input_vector = output
    return outputs

#xor_network = [ [[20, 20, -30],
#                 [20, 20, -10]],
#                [[-60, 60, -30]]]
#for x in [0, 1]:
#    for y in [0, 1]:
#        print x, y, feed_forward(xor_network, [x, y])[-1]


def backpropagate(network, input_vector):
    hidden_outputs, outputs = feed_forward(network, input_vector)

    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(ouputs, targets)]


    for i, ouput_neuron in enumerate(network[-1]):
        for j, hidden output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output


    hidden_deltas = [hidden_output * (1 - hidden_output) *
                     dot(output_deltas, [n[i] for n in output_layer])
                     for i, hidden_output in enumerate(hidden_outputs)]
    
    
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
           hidden_neuron[j] -= hidden_deltas[i] * input


def predict(input):
    return feed_forward(network, input)[-1]

def main():
    random.seed(0)
    input_size = 25
    num_hidden = 5
    output_size = 10
    
    targets = [[1 if i==j else 0 for i in range(10)] for j in range(10)]
    
    hidden_layer = [[random.random() for _ in range(input_size+1)]
                    for _ in range(num_hidden)]

    output_layer = [[random.random() for _ in range(num_hidden + 1)]
                    for _ in range(output_size)]
    
    network = [hidden_layer, output_layer]
    
     

    for _ in range(10000):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)


    

