# nearest neighbor
import random
import math
import numpy


def leave_one_out_cross_validation():
    # uncomment for debugging
    # return random.randint()
    # load data
    data = numpy.loadtxt

    correctly_classified = 0

    for i in range(len(data)):
        object_to_classify = data
        label_object_to_classify = data

        nearest_neighbor_distance = float('Inf')
        nearest_neighbor_location = float('Inf')

        for j in range(len(data)):
            if j != i:
                distance = math.sqrt(sum((object_to_classify - data)**2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = j
                    nearest_neighbor_label = data
        
        if label_object_to_classify == nearest_neighbor_label:
            correctly_classified += 1
        
    return correctly_classified / len(data)



def feature_search(data):
    current_features = set()

    for i in range(len(data)):
        print(f'On the {i}-th level of the search tree')
        feature_to_add = 0
        best_accuracy = 0

        for j in range(len(data)):
            if not current_features.intersection([j]):
                print(f'Considering adding the {j} feature...')
                accuracy = leave_one_out_cross_validation(data, current_features, j+1)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    feature_to_add = j
        
        current_features.add(feature_to_add)
        print(f'On level {i}, feature {feature_to_add} was added to the current set')
