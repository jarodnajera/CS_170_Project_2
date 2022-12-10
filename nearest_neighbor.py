# Original code by Jarod Najera 862179022
# nearest neighbor
import math
import numpy
import copy

# Leave One Out Cross Validation for Forward Elimination
def forward_leave_one_out_cross_validation(data, current_set, feature_to_add):
    # uncomment for debugging
    # return random.randint()

    # configure data to consider certain features
    edited_data = copy.deepcopy(data)
    for row in edited_data:
        if current_set:
            for column, j in enumerate(row):
                if column != 0 and column not in current_set and column != feature_to_add:
                    row[column] = 0
        else:
            for column, j in enumerate(row):
                if column != 0 and column != feature_to_add:
                    row[column] = 0

    number_correctly_classified = 0

    for j, i in enumerate(edited_data):
        object_to_classify = i[1:]
        label_object_to_classify = i[0]


        nearest_neighbor_distance = float('Inf')
        nearest_neighbor_location = float('Inf')

        for l, k in enumerate(edited_data):
            if j != l:
                distance = math.sqrt(sum((object_to_classify - k[1:])**2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = l
                    nearest_neighbor_label = edited_data[l][0]
        
        #print(f'Object {j} is class {label_object_to_classify}')
        #print(f'Its nearest neighbor is {nearest_neighbor_location} which is in class {nearest_neighbor_label}')

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
        
    return number_correctly_classified / len(edited_data)

# Forward Feature Search
def forward_feature_search(data):
    current_set_of_features = set()
    num_features = len(data[0][1:])
    best_features = set()
    best_accuracy = 0

    print('Beginning search...')
    for i in range(num_features):
        #print(f'On the {i+1}-th level of the search tree')
        feature_to_add_at_this_level = 0
        best_so_far_accuracy = 0

        for k in range(num_features):
            if not current_set_of_features.intersection([k+1]):
                #print(f'--Considering adding the {k+1} feature')
                accuracy = forward_leave_one_out_cross_validation(data, current_set_of_features, k+1)
                print(f'Using feature(s) ({current_set_of_features}, {k+1}) accuracy is {accuracy:.3f}')
            
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k+1
                
                    if best_so_far_accuracy > best_accuracy:
                        best_accuracy = copy.deepcopy(best_so_far_accuracy)
                        best_features = copy.deepcopy(current_set_of_features)
                        best_features.add(feature_to_add_at_this_level)


        current_set_of_features.add(feature_to_add_at_this_level)
        print(f'Feature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy:.3f}')
        #print(f'On level {i+1} added feature {feature_to_add_at_this_level} to current set')
    
    print(f'\nFinished search! The best feature subset is {best_features} which has an accuracy of {best_accuracy:.3f}')

# Leave One Out Cross Validation for Backward Elimination
def backward_leave_one_out_cross_validation(data, current_set, feature_to_remove):
    # uncomment for debugging
    # return random.randint()

    # configure data to consider certain features
    edited_data = copy.deepcopy(data)
    for row in edited_data:
        if current_set:
            for column, j in enumerate(row):
                if column != 0 and column not in current_set or column == feature_to_remove:
                    row[column] = 0
        else:
            for column, j in enumerate(row):
                if column != 0 or column == feature_to_remove:
                    row[column] = 0

    number_correctly_classified = 0

    for j, i in enumerate(edited_data):
        object_to_classify = i[1:]
        label_object_to_classify = i[0]


        nearest_neighbor_distance = float('Inf')
        nearest_neighbor_location = float('Inf')

        for l, k in enumerate(edited_data):
            if j != l:
                distance = math.sqrt(sum((object_to_classify - k[1:])**2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = l
                    nearest_neighbor_label = edited_data[l][0]
        
        #print(f'Object {j} is class {label_object_to_classify}')
        #print(f'Its nearest neighbor is {nearest_neighbor_location} which is in class {nearest_neighbor_label}')

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
        
    return number_correctly_classified / len(edited_data)

# Backward Feature Search
def backward_feature_search(data):
    current_set_of_features = set()
    num_features = len(data[0][1:])
    for feature in range(num_features):
        current_set_of_features.add(feature+1)
    best_features = set()
    best_accuracy = 0

    print('Beginning search...')
    for i in range(num_features):
        #print(f'On the {i+1}-th level of the search tree')
        feature_to_remove_at_this_level = 0
        best_so_far_accuracy = 0

        for k in range(num_features):
            if current_set_of_features.intersection([k+1]):
                #print(f'--Considering adding the {k+1} feature')
                accuracy = backward_leave_one_out_cross_validation(data, current_set_of_features, k+1)
                print(f'Using feature(s) ({current_set_of_features} without {k+1}), accuracy is {accuracy:.3f}')
            
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_remove_at_this_level = k+1
                
                    if best_so_far_accuracy > best_accuracy:
                        best_accuracy = copy.deepcopy(best_so_far_accuracy)
                        best_features = copy.deepcopy(current_set_of_features)
                        best_features.remove(feature_to_remove_at_this_level)


        current_set_of_features.remove(feature_to_remove_at_this_level)
        print(f'Feature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy:.3f}')
        #print(f'On level {i+1} added feature {feature_to_add_at_this_level} to current set')
    
    print(f'\nFinished search! The best feature subset is {best_features} which has an accuracy of {best_accuracy:.3f}')
    
# Forward Elimination: forward_feature_search() wrapper
def forward_elimination(data):
    forward_feature_search(data)

# Backward Elimnation: backward_feature_search() wrapper
def backward_elimination(data):
    backward_feature_search(data)