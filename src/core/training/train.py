from core.utils.dataset_operation import clean_data_and_normalize
import copy
import math

HOUSES = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
LEARNING_RATE = 0.0001
EPOCHS = 10000

def prep_data_for_each_house(data, house):
    # data = [row for row in data if row[0] == house row[0] = 1 else row[0] = 0]
    new_data = copy.deepcopy(data)
    for row in new_data:
        if row[0] == house:
            row[0] = 1
        else:
            row[0] = 0
    return new_data


# logistic_function
def sigmoid_function(prediction):
    return (1 / (1 + math.exp(-prediction)))    


import math

def sigmoid_function(z):
    return 1 / (1 + math.exp(-z))


def log_loss(y,):
    pass

def gradient_descend(row, y, weights, learning_rate):
    linear_output = weights[0]  # Bias term
    for element in range(1, len(row)):
        linear_output += ( weights[element] * row[element] )
    
    #sigmoid function to get the prediction

    l_value = sigmoid_function(linear_output) + weights[0]
    
    #error
    error = l_value - y
    
    # update weights (including the bias term)
    weights[0] = weights[0] - learning_rate * error
    for element in range(1, len(row)):
        gradient = error * row[element]
        weights[element] = weights[element] - learning_rate * gradient
    return weights


def train_for_each_house(data):
    model = []
    for house in HOUSES:
        house_data = prep_data_for_each_house(data, house)
        weights = [0.0 for _ in range(len(data[0]))]
        for _ in range(EPOCHS):
            for row in house_data:
                y = row[0]  # Target value
                weights = gradient_descend(row, y, weights, LEARNING_RATE)
                for i in range(len(weights)):
                    weights[i] = weights[i] / len(house_data)
        model.append(weights)
    return model



def train_model(dataset_name):
    array_of_names = ["Index", "First Name", "Last Name", "Birthday", "Best Hand", "Arithmancy", "Care of Magical Creatures"]
    data, header = clean_data_and_normalize(dataset_name, array_of_names)
    model = train_for_each_house(data)
    return [header] + model[:]
