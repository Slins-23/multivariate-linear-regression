import matplotlib.pyplot as plt
import math
import functools
from matplotlib.ticker import FuncFormatter
import random
import time
import numpy as np
import threading
import os

# Uses gradient descent

while True:
    dataset_file_path = input("Dataset file path (relative or absolute): ")
    if os.path.isfile(dataset_file_path):
        break
    else:
        print("Error: Invalid file path. File was not found.")

entries = []
filtered_entries = []
feature_list = []
feature_types = []
feature_columns = []
dependent_variable = ""
dependent_variable_type = ""
dependent_variable_column = -1

nan_values = {}

'''
class Entry:
    def __init__(self, id, property_type, state, region, latitude, longitude, area, price):
        self.id = id
        self.property_type = property_type
        self.state = state
        self.region = region
        self.latitude = latitude
        self.longitude = longitude
        self.area = area
        self.price = price
'''

class Entry:
    def __init__(self, feature_values, dependent_variable_value):
        self.features = {}
        for feature_name, feature_value in zip(feature_list, feature_values):
            self.features[feature_name] = feature_value

        self.dependent_variable_value = dependent_variable_value

def populate_entries():
    global dependent_variable
    global dependent_variable_type
    global dependent_variable_column

    print("Note: The dataset must be .csv formatted, with ',' as a delimiter, and each column in the first row must contain the respective feature name.")
    print("i.e.:")
    print("id,area,price")
    print("0,100,400000")
    print("1,50,250000")
    print("2,200,900000\n")

    with open(dataset_file_path, "r", encoding="utf-8") as dataset:
        rows = dataset.readlines()
        column_names = rows[0].replace("\n", "").split(",")
        print(f"There are {len(column_names)} columns. These are the column names, in ascending column order: {', '.join(column_names)}.")
        column_warning_printed = False
        name_warning_printed = False

        # Breaks loop if all available features have been chosen, as well as the dependent variable
        while len(feature_list) < len(column_names) - 1 or dependent_variable_column == -1 :

            if len(feature_list) > 0 and dependent_variable_column != -1:
                result = input("Select more feature(s) (y/n)? ")
                if result == "n":
                    break
                elif result != "y":
                    print("Invalid input. Must be 'y' or 'n'.")
                    continue

            # Selects a dependent or independent feature by column name or number
            # Does some validation checks (not enough, though)
            while True:
                result = input("Select feature by column name or column number (0 for name / 1 for number)? ")
                if result != "0" and result != "1":
                    print("Invalid input. Must be '0' or '1'.")
                    continue
                
                # Selects by column name
                elif result == "0":
                    if not name_warning_printed:
                        print(f"The column name is the exact string in that column.")
                        name_warning_printed = True

                    name = input("Column name: ")
                    if name not in column_names:
                        print(f"Invalid column name: {name}. Valid column names: {', '.join(column_names)}")
                        continue

                    if name in feature_list:
                        print(f"Error: This column ({name}) is already in the feature list.")
                        continue

                    if name == dependent_variable:
                        print(f"Error: This column ({name}) is already set as the dependent variable.")
                        continue

                    data_type = input("Column data type ('str', 'int', or 'float'): ")
                    if data_type not in ['str', 'int', 'float']:
                        print(f"Invalid data type: {data_type}. Must be one of 'str', 'int', or 'float'.")
                        continue

                    dependent_or_independent = input("Dependent or independent feature (d/i)? ")
                    if dependent_or_independent not in ['d', 'i']:
                        print(f"Invalid feature type: {dependent_or_independent}. Must be 'd' for dependent variable, or 'i' for independent variable.")
                        continue

                    number = column_names.index(name)

                    if dependent_or_independent == 'i':
                        feature_list.append(name)
                        feature_types.append(data_type)
                        feature_columns.append(number)
                    elif dependent_or_independent == 'd':
                        if dependent_variable_column != -1:
                            print(f"Error: Could not set dependent variable to {name}. It is already set to {dependent_variable}")
                            continue

                        dependent_variable = name
                        dependent_variable_type = data_type
                        dependent_variable_column = number

                # Selects by column number
                elif result == "1":
                    # Check whether the name is not unique and if so prompt user for the column number

                    if not column_warning_printed:
                        print("WARNING! FOR THIS TO WORK PROPERLY, EACH COLUMN NAME MUST BE UNIQUE.")
                        print(f"The column number is an integer in the range [0, {len(column_names) - 1}].")
                        print("You cannot choose an already chosen column number, as it leads to redundancy and linear dependence.")
                        print("Exactly one column must the dependent feature.")
                        column_warning_printed = True
                    # add by number

                    number = input("Column number: ")
                    try:
                        number = int(number)
                    except:
                        print(f"Invalid column number: {number}. Must be an integer.")
                        continue

                    if number < 0 or number > len(column_names) - 1:
                        print(f"Invalid column number: {number}. Must be in the range [0,{len(column_names) - 1}]")
                        continue

                    if number in feature_columns:
                        print(f"Error: This column ({number}) is already in the feature list.")
                        continue

                    if number == dependent_variable_column:
                        print(f"Error: This column ({name}) is already set as the dependent variable.")
                        continue

                    name = column_names[number]

                    data_type = input("Column data type ('str', 'int', or 'float'): ")
                    if data_type not in ['str', 'int', 'float']:
                        print(f"Invalid data type {data_type}. Must be one of 'str', 'int', or 'float'.")
                        continue

                    dependent_or_independent = input("Dependent or independent feature (d/i)? ")
                    if dependent_or_independent not in ['d', 'i']:
                        print(f"Invalid feature type: {dependent_or_independent}. Must be 'd' for dependent variable, or 'i' for independent variable.")
                        continue
                    

                    if dependent_or_independent == 'i':
                        feature_list.append(name)
                        feature_types.append(data_type)
                        feature_columns.append(number)
                    elif dependent_or_independent == 'd':
                        if dependent_variable_column != -1:
                            print(f"Error: Could not set dependent variable to {name}. It is already set to {dependent_variable}")
                            continue
                        dependent_variable = name
                        dependent_variable_type = data_type
                        dependent_variable_column = number
                
                break

        print("Finished selecting features.")               

        for line in rows[1:]:
            try:
                columns = line.replace("\n", "").split(",")
                feature_values = []
                for feature_type, feature_column in zip(feature_types, feature_columns):
                    value = columns[feature_column]
                    match feature_type:
                        case "int":
                            value = float(value)
                        case "float":
                            value = float(value)
                        case "str":
                            if value not in nan_values.keys():
                                nan_values[value] = random.randint(1, len(rows))

                            value = nan_values[value]


                    feature_values.append(value)

                dependent_variable_value = columns[dependent_variable_column]
                match dependent_variable_type:
                        case "int":
                            dependent_variable_value = float(dependent_variable_value)
                        case "float":
                            dependent_variable_value = float(dependent_variable_value)
                        case "str":
                            if dependent_variable_value not in nan_values.keys():
                                nan_values[dependent_variable_value] = random.randint(1, len(rows))

                            dependent_variable_value = nan_values[dependent_variable_value]

                '''
                id = int(columns[0])
                property_type = columns[1]
                state = columns[2]
                region = columns[3]
                latitude = float(columns[4])
                longitude = float(columns[5])
                area = int(columns[6])
                price = int(float(columns[7]))

                
                
                new_entry = Entry(id, property_type, state, region, latitude, longitude, area, price)
                '''

                new_entry = Entry(feature_values, dependent_variable_value)
                entries.append(new_entry)
            except:
                continue
        
        dataset.close()

    print("Finished populating the entries.")

def custom_filter(entry, target_features, comparison_targets, comparison_operators):
    for feature, target, operator in zip(target_features, comparison_targets, comparison_operators):
        cmp = True
        match operator:
            case '==':
                if feature == dependent_variable:
                    cmp = entry.dependent_variable_value == target
                else:
                    cmp = entry.features[feature] == target
            case '!=':
                if feature == dependent_variable:
                    cmp = entry.dependent_variable_value != target
                else:
                    cmp = entry.features[feature] != target
            case '<':
                if feature == dependent_variable:
                    cmp = entry.dependent_variable_value < target
                else:
                    cmp = entry.features[feature] < target
            case '<=':
                if feature == dependent_variable:
                    cmp = entry.dependent_variable_value <= target
                else:
                    cmp = entry.features[feature] <= target
            case '>=':
                if feature == dependent_variable:
                    cmp = entry.dependent_variable_value >= target
                else:
                    cmp = entry.features[feature] >= target
            case '>':
                if feature == dependent_variable:
                    cmp = entry.dependent_variable_value > target
                else:
                    cmp = entry.features[feature] > target

        if not cmp:
            return False

    return True

def filter_entries(should_filter, filtered_list):
    if should_filter:
        print("Note: The filtering occurs only once, for all features.")

        valid_operators = ["==", "!=", "<", "<=", ">=", ">"]
        target_features = []
        comparison_targets = []
        comparison_operators = []
        while True:
            l_target_features = input(f"Which features to filter (comma-separated)? Options: {', '.join(feature_list)}, {dependent_variable} ")
            l_target_features = l_target_features.split(",")
            feature_not_found = False

            for target_feature in l_target_features:
                if target_feature not in feature_list and target_feature != dependent_variable:
                    print(f"Feature {target_feature} is not a valid feature.")
                    feature_not_found = True
                    break

            if feature_not_found:
                continue

            l_comparison_targets = []
            target_invalid = False
            

            # These checks are separated in different loops (resulting in 3 times the necessary time for execution)
            # The reason is for more clarity and isolation of validation checks
            # It shouldn't slow down most times, as there usually aren't that many features (at least with what I have in mind)
            # Also, it is not optimized and I could definitely write this in a more isolated, clear, way, with faster runtimes as well
            for target_feature in l_target_features:
                if target_feature != dependent_variable:
                    feature_index = feature_list.index(target_feature)
                    feature_type = feature_types[feature_index]

                    comparison_target = input(f"Type in the target value to compare the feature {target_feature} against. Your input must be a valid '{feature_type}'! ")

                    if feature_type == "str" and comparison_target not in nan_values.keys():
                        print("Error: Invalid target value. Given value is not in the dataset.")
                        target_invalid = True
                        break
                    elif feature_type == "int" or feature_type == "float":
                        try:
                            comparison_target = float(comparison_target)
                        except:
                            print("Error: Invalid target value. Could not convert it to a number.")
                            target_invalid = True
                            break

                    l_comparison_targets.append(comparison_target)
                else:
                    comparison_target = input(f"Type in the target value to compare the feature {target_feature} against. Your input must be a valid '{dependent_variable_type}'! ")

                    if dependent_variable_type == "str" and comparison_target not in nan_values.keys():
                        print("Error: Invalid target value. Given value is not in the dataset.")
                        target_invalid = True
                        break
                    elif dependent_variable_type == "int" or dependent_variable_type == "float":
                        try:
                            comparison_target = float(comparison_target)
                        except:
                            print("Error: Invalid target value. Could not convert it to a number.")
                            target_invalid = True
                            break

                    l_comparison_targets.append(comparison_target)


            if target_invalid:
                continue

            l_comparison_operators = []
            invalid_operator = False

            for target_feature in l_target_features:
                if target_feature != dependent_variable:
                    feature_index = feature_list.index(target_feature)
                    feature_type = feature_types[feature_index]
                
                    comparison_operator = input(f"Type in the comparison operator (must be '==', '!=', '<', '<=', '>=' or '>'): ").strip()
                    
                    if comparison_operator not in valid_operators:
                        print("Error: Given comparison operator is not in the list.")
                        invalid_operator = True
                        break

                    if feature_type == "str":
                        if comparison_operator != "==" and comparison_operator != "!=":
                            print("Error: Strings can only be compared using the equality operators '==' and '!='.")
                            invalid_operator = True
                            break

                    l_comparison_operators.append(comparison_operator)
                else:
                    comparison_operator = input(f"Type in the comparison operator (must be '==', '!=', '<', '<=', '>=' or '>'): ").strip()
                    
                    if comparison_operator not in valid_operators:
                        print("Error: Given comparison operator is not in the list.")
                        invalid_operator = True
                        break

                    if dependent_variable_type == "str":
                        if comparison_operator != "==" and comparison_operator != "!=":
                            print("Error: Strings can only be compared using the equality operators '==' and '!='.")
                            invalid_operator = True
                            break

                    l_comparison_operators.append(comparison_operator)

            if invalid_operator:
                continue

            target_features = l_target_features
            comparison_targets = l_comparison_targets
            comparison_operators = l_comparison_operators

            break

        filtered_list = list(filter(lambda entry: custom_filter(entry, target_features, comparison_targets, comparison_operators), filtered_list))
            

    print("Samples have been filtered.")

    return filtered_list

'''
def filter_entries(property_type=None, state=None, region=None, min_area=None, max_area=None, min_price=None, max_price=None):
    filtered_list = entries

    conditions = [
        (property_type, lambda entry: entry.property_type == property_type),
        (state, lambda entry: entry.state == state),
        (region, lambda entry: entry.region == region),
        (min_area, lambda entry: entry.area >= min_area),
        (max_area, lambda entry: entry.area <= max_area),
        (min_price, lambda entry: entry.price >= min_price),
        (max_price, lambda entry: entry.price <= max_price),
                  ]
    
    for property, condition in conditions:
        if property != None and property != False:
            filtered_list = list(filter(condition, filtered_list))

    return filtered_list
'''

def string_to_int(string):
    if string in nan_values.keys():
        return nan_values[string]
    else:
        raise(f"Error converting string to integer. Feature string '{string}' not found in the dataset.")

'''
def conv_property_type(property_type):
    if property_type == "house":
        return 0
    elif property_type == "apartment":
        return 1
    else:
        raise("Invalid property type! Must be 'house' or 'apartment'.")
    
def conv_state(state):
    if state == "Pernambuco":
        return 0
    elif state == "Rio de Janeiro":
        return 1
    elif state == "São Paulo":
        return 2
    elif state == "Piauí":
        return 3
    elif state == "Rio Grande do Sul":
        return 4
    elif state == "Tocantins":
        return 5
    elif state == "Santa Catarina":
        return 6
    elif state == "Sergipe":
        return 7
    elif state == "Rio Grande do Norte":
        return 8
    elif state == "Rondônia":
        return 9
    else:
        raise("Invalid state! Must be 'Pernambuco', 'Rio de Janeiro', 'São Paulo', 'Piauí', 'Rio Grande do Sul', 'Tocantins', 'Santa Catarina', 'Sergipe', 'Rio Grande do Norte' or 'Rondônia'.")
    
def conv_region(region):
    if region == "North":
        return 0
    elif region == "Northeast":
        return 1
    elif region == "Southeast":
        return 2
    elif region == "South":
        return 3
    else:
        raise("Invalid region! Must be 'North', 'Northeast', 'Southeast' or 'South'.")
'''
        
def normalize_helper(prev_value, prev_min, prev_max, new_min, new_max):
    prev_value_range = prev_max - prev_min
    percentage_of_previous_range = (prev_value - prev_min) / prev_value_range
    new_value_range = new_max - new_min
    
    return new_min + (new_value_range * percentage_of_previous_range)

def normalize(entry):
    # Ignores last feature (the dependent variable y to be predicted)
    for feature, idx in zip(feature_list, range(K)):
        entry.features[feature] = normalize_helper(
            entry.features[feature],
            feature_min_maxes[idx][0],
            feature_min_maxes[idx][1],
            0,
            1
            )

    '''
    entry.property_type = normalize_helper(
        entry.property_type,
        feature_min_maxes[0][0],
        feature_min_maxes[0][1],
        0,
        1
        )
    
    entry.state = normalize_helper(
        entry.state,
        feature_min_maxes[1][0],
        feature_min_maxes[1][1],
        0,
        1
        )
    
    entry.region = normalize_helper(
        entry.region,
        feature_min_maxes[2][0],
        feature_min_maxes[2][1],
        0,
        1
        )
    
    entry.latitude = normalize_helper(
        entry.latitude,
        feature_min_maxes[3][0],
        feature_min_maxes[3][1],
        0,
        1
        )
    
    entry.longitude = normalize_helper(
        entry.longitude,
        feature_min_maxes[4][0],
        feature_min_maxes[4][1],
        0,
        1
        )
    
    entry.area = normalize_helper(
        entry.area,
        feature_min_maxes[5][0],
        feature_min_maxes[5][1],
        0,
        1
        )
    '''
    
def expected_fn(entry):
    # vally = B0 + (B1 * entry.property_type) + (B2 * entry.state) + (B3 * entry.region) + (B4 * entry.latitude) + (B5 * entry.longitude) + (B6 * entry.area)
    #vally = B0 + (B1 * entry.property_type) + (B2 * entry.state) + (B6 * entry.area)
    # print(vally)
    # print(f"B0: {B0} | B1: {B1} | B2: {B2} | B3: {B3} | B4: {B4} | B5: {B5} | B6: {B6}")
    # print(f"property_type: {entry.property_type} | state: {entry.state} | region: {entry.region} | latitude: {entry.latitude} | longitude: {entry.longitude} | area: {entry.area}")

    # Matrix as NDarray
    X_i = np.full((K + 1, 1), 1, dtype=np.float64)

    # Matrix as Matrix
    X_i = np.matrix(X_i)

    for feature, row in zip(feature_list, range(1, K + 1)):
        X_i[row, 0] = entry.features[feature]


    '''
    X_i = np.matrix(
        [[1, entry.property_type, entry.state, entry.region, entry.latitude, entry.longitude, entry.area]],
        dtype=np.float64
        )
    '''

    inner_product = B.transpose() * X_i
    vally = inner_product[0, 0]
    return vally

def minimize(l_entries):
    '''
    global B0
    global B1
    global B2
    global B3
    global B4
    global B5
    global B6
    global partial_b0
    global partial_b1
    global partial_b2
    global partial_b3
    global partial_b4
    global partial_b5
    global partial_b6
    '''
    global B
    global P

    # expected_fn = lambda x_i: ((M * x_i) + B)

    # Matrix as NDarray
    P = np.zeros((B.shape[0], 1), dtype=np.float64)

    # Matrix as matrix
    P = np.matrix(P)
    '''
    sum_b0 = 0
    sum_b1 = 0
    sum_b2 = 0
    sum_b3 = 0
    sum_b4 = 0
    sum_b5 = 0
    sum_b6 = 0
    '''

    mse = 0

    '''
    Matrix of expected values

                                [ 1    1  . . .   1 ]
                                [ x11 x22 . . . x2n ]
                                [ x21 x32 . . . x3n ]
    [ B0 B1 B2 . . . BK]   *    [ .    .  . . .   . ] =   [Ex(X_1) Ex(X_2) . . . Ex(X_N)]
                                [ .    .  . . .   . ]
                                [ .    .  . . .   . ]
                                [ xk1 xk2 . . . xkn ]
    '''
    EXPECTED_VALUES_MATRIX = B.transpose() * X
    EXPECTED_VALUES_MATRIX = EXPECTED_VALUES_MATRIX.transpose()
    ERROR_MATRIX = EXPECTED_VALUES_MATRIX - TARGET_VALUES_MATRIX

    '''
    for entry, i in zip(l_entries, range(N)):
        tmp = np.matrix([
            [expected_fn(entry) - entry.price],
            [entry.property_type * (expected_fn(entry) - entry.price)],
            [entry.state * (expected_fn(entry) - entry.price)],
            [entry.region * (expected_fn(entry) - entry.price)],
            [entry.latitude * (expected_fn(entry) - entry.price)],
            [entry.longitude * (expected_fn(entry) - entry.price)],
            [entry.area * (expected_fn(entry) - entry.price)]
        ])

        P += tmp
        
        sum_b0 += expected_fn(entry.property_type, entry.state, entry.region, entry.latitude, entry.longitude, entry.area) - entry.price
        sum_b1 += entry.property_type * (expected_fn(entry.property_type, entry.state, entry.region, entry.latitude, entry.longitude, entry.area) - entry.price)
        sum_b2 += entry.state * (expected_fn(entry.property_type, entry.state, entry.region, entry.latitude, entry.longitude, entry.area) - entry.price)
        sum_b3 += entry.region * (expected_fn(entry.property_type, entry.state, entry.region, entry.latitude, entry.longitude, entry.area) - entry.price)
        sum_b4 += entry.latitude * (expected_fn(entry.property_type, entry.state, entry.region, entry.latitude, entry.longitude, entry.area) - entry.price)
        sum_b5 += entry.longitude * (expected_fn(entry.property_type, entry.state, entry.region, entry.latitude, entry.longitude, entry.area) - entry.price)
        sum_b6 += entry.area * (expected_fn(entry.property_type, entry.state, entry.region, entry.latitude, entry.longitude, entry.area) - entry.price)

        mse += math.pow((expected_fn(entry) - entry.price), 2)
    '''
    P = X * ERROR_MATRIX
    P /= N
    '''
    partial_b0 = sum_b0 / N
    partial_b1 = sum_b1 / N
    partial_b2 = sum_b2 / N
    partial_b3 = sum_b3 / N
    partial_b4 = sum_b4 / N
    partial_b5 = sum_b5 / N
    partial_b6 = sum_b6 / N
    '''

    mse = (ERROR_MATRIX.transpose() * ERROR_MATRIX)[0, 0]
    iteration_loss = mse / (2 * N)
    iteration_loss = round(iteration_loss)
    loss.append(iteration_loss)

    # print(f"M: {M} | PM: {partial_m} -> {learning_rate * partial_m} | B: {B} | PB: {partial_b} -> {learning_rate * partial_b}")
    # print(f"M: {M} | Partial: {partial_m} | Change: {m_learning_rate * partial_m}")
    # print(f"B: {B} | Partial: {partial_b} | Change: {b_learning_rate * partial_b}")
    
    B -= LR * P
    '''
    B0 -= LR * partial_b0
    B1 -= LR * partial_b1
    B2 -= LR * partial_b2
    B3 -= LR * partial_b3
    B4 -= LR * partial_b4
    B5 -= LR * partial_b5
    B6 -= LR * partial_b6
    '''
    
    # M = round(M)
    # B = round(B)

def train(l_entries, total_steps=1000):
    FPS = 60
    animation_duration_seconds = total_steps / FPS
    ms_per_frame = 1000 / FPS
    iteration = 1
    for step in range(total_steps):
        minimize(l_entries)
        parameters.append(B)
        # print(f"M: {M}, B: {B}, MSE: {loss}")

def predict():
    while True:
        feature_values = []
        for feature, feature_type in zip(feature_list, feature_types):
            while True:
                value = input(f"{feature}: ")

                if feature_type == "str":
                    if value not in nan_values.keys():
                        print("Error: Invalid feature value. Given value is not in the dataset.")
                        continue
                    else:
                        value = string_to_int(value)
                elif feature_type == "int" or feature_type == "float":
                    try:
                        value = float(value)
                    except:
                        print(f"Error: Value {value} is not a valid number.")
                        continue

                feature_values.append(value)
                break

        '''
        property_type = input("Property type: ").strip()
        try:
            property_type = conv_property_type(property_type)
        except Exception as e:
            print(e)
            continue

        state = input("State: ").strip()
        try:
            state = conv_state(state)
        except Exception as e:
            print(e)
            continue

        region = input("Region: ").strip()
        try:
            region = conv_region(region)
        except Exception as e:
            print(e)
            continue
        
        
        latitude = input("Latitude: ").strip()
        try:
            latitude = float(latitude)
        except:
            print("Invalid latitude! Must be a float.")
            continue

        longitude = input("Longitude: ").strip()
        try:
            longitude = float(longitude)
        except:
            print("Invalid longitude! Must be a float.")
            continue
        

        area = input("Area (m²): ").strip()
        try:
            area = int(area)
        except:
            print("Invalid area! Must be an integer.")
            continue

        # entry = Entry(-1, property_type, state, region, latitude, longitude, area, -1)
        entry = Entry(-1, property_type, state, -1, -1, -1, area, -1)
        '''
        entry = Entry(feature_values, -1)

        normalize(entry)

        predicted_value = expected_fn(entry)

        # Need to implement/fix script for discrete dependent variables (i.e. wanting to predict a string, an integer in a specific range like 0 or 1, etc...)
        '''

        # Will not work properly if more than one string in the dataset's training samples are represented by the same number
        if dependent_variable_type == "str":
            for value, key in zip(nan_values.values(), nan_values.keys()):
                if value == predicted_value:
                    predicted_value = key
                    predicted_value = normalize_helper(predicted_value, )
                    break
        '''

        print(f"Predicted value: {predicted_value}")

populate_entries()

print("Dataset loaded.")

filtered_entries = entries
while True:
    should_filter = input("Filter samples (y/n)? ")
    if should_filter == "y":
        should_filter = True
    elif should_filter == "n":
        should_filter = False
    else:
        print("Invalid input. Should be 'y' or 'n' without the quotation marks.")
        continue

    filtered_entries = filter_entries(should_filter, filtered_entries)
    if not should_filter: break
    
print(f"Total samples: {len(entries)}")
print(f"Total filtered samples: {len(filtered_entries)}")

TOTAL_STEPS = 50
while True:
    given_steps = input("How many training steps? (must be a number) ")
    try:
        TOTAL_STEPS = int(given_steps)
    except:
        print("Error: Could not convert given training steps to an integer.")
        continue

    break

K = len(feature_list)

# feature_min_maxes = [[0, -math.inf], [0, -math.inf], [0, -math.inf], [math.inf, -math.inf], [math.inf, -math.inf], [math.inf, -math.inf]]
feature_min_maxes = []
for feature_type in feature_types:
    if feature_type == "str":
        feature_min_maxes.append([0, -math.inf])
    elif feature_type == "float" or feature_type == "int":
        feature_min_maxes.append([math.inf, -math.inf])

for entry in filtered_entries:
    '''
    entry.property_type = conv_property_type(entry.property_type)
    entry.state = conv_state(entry.state)
    entry.region = conv_region(entry.region)
    '''

    for feature, idx in zip(feature_list, range(K)):
        feature_min_maxes[idx][0] = min(entry.features[feature], feature_min_maxes[idx][0])
        feature_min_maxes[idx][1] = max(entry.features[feature], feature_min_maxes[idx][1])

    '''
    feature_min_maxes[0][0] = min(entry.property_type, feature_min_maxes[0][0])
    feature_min_maxes[0][1] = max(entry.property_type, feature_min_maxes[0][1])

    feature_min_maxes[1][0] = min(entry.state, feature_min_maxes[1][0])
    feature_min_maxes[1][1] = max(entry.state, feature_min_maxes[1][1])

    feature_min_maxes[2][0] = min(entry.region, feature_min_maxes[2][0])
    feature_min_maxes[2][1] = max(entry.region, feature_min_maxes[2][1])

    feature_min_maxes[3][0] = min(entry.latitude, feature_min_maxes[3][0])
    feature_min_maxes[3][1] = max(entry.latitude, feature_min_maxes[3][1])

    feature_min_maxes[4][0] = min(entry.longitude, feature_min_maxes[4][0])
    feature_min_maxes[4][1] = max(entry.longitude, feature_min_maxes[4][1])

    feature_min_maxes[5][0] = min(entry.area, feature_min_maxes[5][0])
    feature_min_maxes[5][1] = max(entry.area, feature_min_maxes[5][1])
    '''

N = len(filtered_entries)

for entry in filtered_entries:
    normalize(entry)

'''

            Input Matrix
        ---------------------
    Each entry is a column vector
        (Numpy indexes at 0)
        k = Number of features
        n = Number of entries
              (k + 1) x n
        [ 1    1  . . .   1 ]
        [ x11 x22 . . . x2n ]
        [ x21 x32 . . . x3n ]
        [ .    .  . . .   . ]
        [ .    .  . . .   . ]
        [ .    .  . . .   . ]
        [ xk1 xk2 . . . xkn ]

'''

# Matrix as NDarray
X = np.full((K + 1, N), 1, dtype=np.float64)

# Matrix as Matrix
X = np.matrix(X)

'''
            Output Matrix
'''

# Matrix as NDArray
TARGET_VALUES_MATRIX = np.zeros((N, 1), dtype=np.float64)

# Matrix as Matrix
TARGET_VALUES_MATRIX = np.matrix(TARGET_VALUES_MATRIX)

for entry, col in zip(filtered_entries, range(N)):
    TARGET_VALUES_MATRIX[col, 0] = entry.dependent_variable_value
    for feature, row in zip(feature_list, range(1, K + 1)):
        X[row, col] = entry.features[feature]



# exit(-1)

# Weight matrix
# B = np.matrix(np.arange((K + 1) * 1).reshape((K + 1, 1)), dtype=np.float64)

# Matrix as NDarray
B = np.zeros((K + 1, 1), dtype=np.float64)

# Matrix as Matrix
B = np.matrix(B)

for row in range(K + 1):
    B[row, 0] = random.randint(1, 10)

'''
B = np.matrix([
    [random.randint()],
    [random.randint()],
    [random.randint()],
    [random.randint()],
    [random.randint()],
    [random.randint()],
    [random.randint()]
], dtype=np.float64)
'''

# Partials matrix
# Matrix as NDarray
P = np.zeros((K + 1, 1), dtype=np.float64)

# Matrix as Matrix
P = np.matrix(P)

# P = np.matrix(np.arange((K + 1) * 1).reshape((K + 1, 1)), dtype=np.float64)
for row in range(K + 1):
    P[row, 0] = random.randint(1, 10)

# P = np.matrix([[0], [0], [0], [0], [0], [0], [0]], dtype=np.float64)

'''
B0 = -9999
B1 = 3232323
B2 = 333
B3 = -32.33
B4 = 231.4
B5 = 200.1
B6 = -3

partial_b0 = 0
partial_b1 = 0
partial_b2 = 0
partial_b3 = 0
partial_b4 = 0
partial_b5 = 0
partial_b6 = 0
'''

LR = 0.5
while True:
    given_lr = input("Learning rate (must be a number): ")
    try:
        LR = float(given_lr)
    except:
        print("Error: Could not convert given learning rate to a float.")
        continue

    break


# parameters = [[B0, B1, B2, B3, B4, B5, B6]]
parameters = [B]
loss = []
iterations = [iteration + 1 for iteration in range(TOTAL_STEPS)]

train(filtered_entries, TOTAL_STEPS)

plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.plot(iterations, loss)

thread = threading.Thread(target=predict)
thread.start()

plt.show()

thread.join()

