import numpy as np

input_features = np.array([[0,0],[0,1],[1,0],[1,1]])
print(input_features.shape)
input_features

target_output = np.array([[0,1,1,1]])
target_output = target_output.reshape(4,1)

print(target_output.shape)
target_output

weights = np.array([[0.1],[0.2]])
print(weights.shape)
weights

bias = 0.3

lr = 0.05

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

for epoch in range(10000):
    print("output at epoch: " + str(epoch))

    inputs = input_features

    in_o = np.dot(inputs, weights) + bias

    out_o = sigmoid(in_o)
    print(out_o)
    error = out_o - target_output

    x = error.sum()
    #print(x)

    derror_douto = error
    douto_dino = sigmoid_der(out_o)

    deriv = derror_douto * douto_dino
    inputs = input_features.T
    deriv_final = np.dot(inputs, deriv)

    weights -= lr * deriv_final

    for i in deriv:
        bias -= lr * i

    #print(weights)
    #print(bias)

    single_point = np.array([0,0])

    result1 = np.dot(single_point, weights) + bias

    result2 = sigmoid(result1)
    '''
      print(result2)

    single_point = np.array([1,1])

    result1 = np.dot(single_point, weights) + bias

    result2 = sigmoid(result1)

    print(result2)

    single_point = np.array([0, 0])

    result1 = np.dot(single_point, weights) + bias

    result2 = sigmoid(result1)

    print(result2)

    '''


