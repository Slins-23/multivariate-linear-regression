from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import math
import functools
from matplotlib.ticker import FuncFormatter
from matplotlib.animation import FuncAnimation
import random
import time
import numpy as np
import threading
import os

while True:
    dataset_file_path = input("Dataset file path (relative or absolute): ")
    if os.path.isfile(dataset_file_path):
        break
    else:
        print("Error: Invalid file path. File was not found.")

entries = []
columns = []
column_types = []
filtered_entries = []
feature_list = []
feature_types = []
feature_columns = []
dependent_variable = ""
dependent_variable_type = ""
dependent_variable_column = -1

nan_values = {}

reference_features = []
reference_values = []
reference_types = []
min_distance = math.inf
max_distance = -math.inf

class Entry:
    def __init__(self, feature_values, dependent_variable_value, column_value_dict):
        self.features = {}
        for feature_name, feature_value in zip(feature_list, feature_values):
            self.features[feature_name] = feature_value

        self.dependent_variable_value = dependent_variable_value

        self.column_value_dict = column_value_dict

        self.distance = None


    # Euclidean distance
    def distance_from_reference(self):
        if reference_values == [] or reference_features == []:
            return None
        
        dist = 0
        for reference_feature, reference_value in zip(reference_features, reference_values):
            if reference_feature in feature_list:
                current_feature_value = self.features[reference_feature]

            elif reference_feature == dependent_variable:
                current_feature_value = self.dependent_variable_value

            elif reference_feature in columns:
                current_feature_value = self.column_value_dict[reference_feature]
            else:
                print(f"Error: Reference feature '{reference_feature}' is not in the dataset.")
                exit(-1)
                
                
            dist += math.pow(reference_value - current_feature_value, 2)

        dist = math.sqrt(dist)

        return dist

def populate_entries():
    global columns
    global column_types
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
        columns = column_names
        column_types = [None for i in range(len(columns))]
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
                        print("Exactly one column must be the dependent feature.")
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
                entry_columns = line.replace("\n", "").split(",")
                feature_values = []
                for feature_type, feature_column in zip(feature_types, feature_columns):
                    value = entry_columns[feature_column]
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

                dependent_variable_value = entry_columns[dependent_variable_column]
                match dependent_variable_type:
                        case "int":
                            dependent_variable_value = float(dependent_variable_value)
                        case "float":
                            dependent_variable_value = float(dependent_variable_value)
                        case "str":
                            if dependent_variable_value not in nan_values.keys():
                                nan_values[dependent_variable_value] = random.randint(1, len(rows))

                            dependent_variable_value = nan_values[dependent_variable_value]


                # Note: All columns which are not set as features, are set to be filtered, and represent a number, will be treated as a number.
                # In other words. If the column is equal to 1 or "1", the script will always treat this as the number 1.
                column_value_dict = {}
                for column_name, idx in zip(column_names, range(len(column_names))):
                    column_value = entry_columns[idx]
                    column_types[idx] = 'str'
                    try:
                        column_value = float(column_value)
                        column_types[idx] = 'float'
                    except:
                        pass
                    column_value_dict[column_name] = column_value

                new_entry = Entry(feature_values, dependent_variable_value, column_value_dict)
                entries.append(new_entry)
            except:
                continue
        
        dataset.close()

    print("Finished populating the entries.")

def custom_filter(entry, target_features, comparison_targets, comparison_operators):
    global min_distance
    global max_distance

    if reference_values != []:
        if entry.distance == None:
            entry.distance = entry.distance_from_reference()

            if entry.distance > max_distance:
                max_distance = entry.distance
            elif entry.distance < min_distance:
                min_distance = entry.distance

        

    for feature, target, operator in zip(target_features, comparison_targets, comparison_operators):
        cmp = True

        if feature != "distance_builtin":
            match operator:
                case '==':
                    if feature == dependent_variable:
                        cmp = entry.dependent_variable_value == target
                    elif feature in feature_list:
                        cmp = entry.features[feature] == target
                    else:
                        cmp = entry.column_value_dict[feature] == target
                case '!=':
                    if feature == dependent_variable:
                        cmp = entry.dependent_variable_value != target
                    elif feature in feature_list:
                        cmp = entry.features[feature] != target
                    else:
                        cmp = entry.column_value_dict[feature] != target
                case '<':
                    if feature == dependent_variable:
                        cmp = entry.dependent_variable_value < target
                    elif feature in feature_list:
                        cmp = entry.features[feature] < target
                    else:
                        cmp = entry.column_value_dict[feature] < target
                case '<=':
                    if feature == dependent_variable:
                        cmp = entry.dependent_variable_value <= target
                    elif feature in feature_list:
                        cmp = entry.features[feature] <= target
                    else:
                        cmp = entry.column_value_dict[feature] <= target
                case '>=':
                    if feature == dependent_variable:
                        cmp = entry.dependent_variable_value >= target
                    elif feature in feature_list:
                        cmp = entry.features[feature] >= target
                    else:
                        cmp = entry.column_value_dict[feature] >= target
                case '>':
                    if feature == dependent_variable:
                        cmp = entry.dependent_variable_value > target
                    elif feature in feature_list:
                        cmp = entry.features[feature] > target
                    else:
                        cmp = entry.column_value_dict[feature] > target
        elif feature == "distance_builtin":
            match operator:
                case '==':
                    cmp = entry.distance == target
                case '!=':
                    cmp = entry.distance != target
                case '<':
                    cmp = entry.distance < target
                case '<=':
                    cmp = entry.distance <= target
                case '>=':
                    cmp = entry.distance >= target
                case '>':
                    cmp = entry.distance > target

        if not cmp:
            return False

    return True

def filter_entries(should_filter, filtered_list):
    global min_distance
    global max_distance

    if should_filter:
        print("Note: The filtering occurs only once, for all features.")

        valid_operators = ["==", "!=", "<", "<=", ">=", ">"]
        target_features = []
        comparison_targets = []
        comparison_operators = []
        while True:
            if reference_features == []:
                l_target_features = input(f"Which features/columns to filter (comma-separated and no whitespace i.e. 'area_m2,price_brl')? Options: {', '.join(columns)} ")
            else:
                l_target_features = input(f"Which features/columns to filter (comma-separated and no whitespace i.e. 'area_m2,price_brl')? Options: {', '.join(columns)}, distance_builtin ")

            l_target_features = l_target_features.split(",")

            l_comparison_targets = []
            target_invalid = False
            
            # These checks are separated in different loops (resulting in 3 times the necessary time for execution)
            # The reason is for more clarity and isolation of validation checks
            # It shouldn't slow down most times, as there usually aren't that many features (at least with what I have in mind)
            # Also, it is not optimized and I could definitely write this in a more isolated, clear, way, with faster runtimes as well
            for target_feature in l_target_features:
                if target_feature in feature_list:
                    feature_index = feature_list.index(target_feature)
                    feature_type = feature_types[feature_index]

                    comparison_target = input(f"Type in the target value to compare the column '{target_feature}' against. Your input must be a valid '{feature_type}'! ")

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
                elif target_feature == dependent_variable:
                    comparison_target = input(f"Type in the target value to compare the column '{target_feature}' against. Your input must be a valid '{dependent_variable_type}'! ")

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

                elif target_feature == "distance_builtin":
                    comparison_target = input(f"Type in the target value to compare the euclidean distance against the provided reference values. Your input must be a valid number! ")
                    try:
                        comparison_target = float(comparison_target)
                    except:
                        print("Error: Invalid target value. Could not convert it to a number.")
                        target_invalid = True
                        break

                    l_comparison_targets.append(comparison_target)
                else:
                    column_index = columns.index(target_feature)
                    column_type = column_types[column_index]

                    comparison_target = input(f"Type in the target value to compare the column '{target_feature}' against. Your input must be a valid '{column_type}'! ")

                    if column_type == "int" or column_type == "float":
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

            for idx, target_feature in enumerate(l_target_features):
                if target_feature in feature_list:
                    feature_index = feature_list.index(target_feature)
                    feature_type = feature_types[feature_index]
                
                    comparison_operator = input(f"Type in the comparison operator to compare the column '{target_feature}' against the value '{l_comparison_targets[idx]}' (must be '==', '!=', '<', '<=', '>=' or '>'): ").strip()
                    
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
                elif target_feature == dependent_variable:
                    comparison_operator = input(f"Type in the comparison operator to compare the column '{target_feature}' against the value '{l_comparison_targets[idx]}' (must be '==', '!=', '<', '<=', '>=' or '>'): ").strip()
                    
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
                elif target_feature == "distance_builtin":
                    comparison_operator = input(f"Type in the comparison operator to compare the distance against the value '{l_comparison_targets[idx]}' (must be '==', '!=', '<', '<=', '>=' or '>'): ").strip()
                    
                    if comparison_operator not in valid_operators:
                        print("Error: Given comparison operator is not in the list.")
                        invalid_operator = True
                        break

                    l_comparison_operators.append(comparison_operator)
                else:
                    column_index = columns.index(target_feature)
                    column_type = column_types[column_index]
                
                    comparison_operator = input(f"Type in the comparison operator to compare the column '{target_feature}' against the value '{l_comparison_targets[idx]}' (must be '==', '!=', '<', '<=', '>=' or '>'): ").strip()
                    
                    if comparison_operator not in valid_operators:
                        print("Error: Given comparison operator is not in the list.")
                        invalid_operator = True
                        break

                    if column_type == "str":
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

        if reference_features != []:
            min_distance = math.inf
            max_distance = -math.inf

        filtered_list = list(filter(lambda entry: custom_filter(entry, target_features, comparison_targets, comparison_operators), filtered_list))
    else:
        if reference_features != []:
            min_distance = math.inf
            max_distance = -math.inf

            for entry in filtered_list:
                if entry.distance == None:
                    entry.distance = entry.distance_from_reference()

                if entry.distance > max_distance:
                    max_distance = entry.distance
                elif entry.distance < min_distance:
                    min_distance = entry.distance
            

    print("Samples have been filtered.")

    return filtered_list

def string_to_int(string):
    if string in nan_values.keys():
        return nan_values[string]
    else:
        raise(f"Error converting string to integer. Feature string '{string}' not found in the dataset.")
        
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
    
def expected_fn(entry):
    # Matrix as NDarray
    X_i = np.full((K + 1, 1), 1, dtype=np.float64)

    # Matrix as Matrix
    X_i = np.matrix(X_i)

    for feature, row in zip(feature_list, range(1, K + 1)):
        X_i[row, 0] = entry.features[feature]

    inner_product = B.transpose() * X_i
    return inner_product[0, 0]

def minimize(l_entries, iteration):
    global B
    global P

    # Matrix as NDarray
    P = np.zeros((B.shape[0], 1), dtype=np.float64)

    # Matrix as matrix
    P = np.matrix(P)

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

    P = X * ERROR_MATRIX
    P /= N

    mse = (ERROR_MATRIX.transpose() * ERROR_MATRIX)[0, 0]
    iteration_loss = mse / (2 * N)
    # loss[iteration - 1] = iteration_loss
    loss.append(iteration_loss)

    B -= LR * P

    # print(f"M: {M} | PM: {partial_m} -> {learning_rate * partial_m} | B: {B} | PB: {partial_b} -> {learning_rate * partial_b}")
    # print(f"M: {M} | Partial: {partial_m} | Change: {m_learning_rate * partial_m}")
    # print(f"B: {B} | Partial: {partial_b} | Change: {b_learning_rate * partial_b}")

def train(l_entries, total_steps=1000):
    global finished

    # plt.show()
    FPS = 60
    animation_duration_seconds = total_steps / FPS
    ms_per_frame = 1000 / FPS
    iteration = 1
    for step in range(total_steps):
        minimize(l_entries, iteration)
        parameters.append(B)
        # plt.plot(iterations, loss)
        iterations.append(iteration)
        iteration += 1
        # print(f"M: {M}, B: {B}, MSE: {loss}")

    finished = True
    print("Done training model.")
    print(f"Final weights: {B}")
    print(f"Final partials: {P}")

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

        entry = Entry(feature_values, -1, {})

        if should_normalize:
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

        print(f"Predicted value ({dependent_variable}): {predicted_value}")

populate_entries()

print("Dataset loaded.")

while True:
    result = input("Give reference value? (y/n) ")
    if result == 'y':
        while True:
            print("Must be a number!")
            l_given_reference_features = input(f"Which feature(s)/column(s) to use as reference(s)? (comma-separated and no whitespace i.e. 'area_m2,price_brl')? Options: {', '.join(columns)} ")
            l_given_reference_features = l_given_reference_features.split(",")

            l_reference_features = []
            l_reference_types = []
            reference_feature_invalid = False
            for reference_feature in l_given_reference_features:
                if reference_feature in l_reference_features:
                    print(f"Error: Duplicate feature/column '{reference_feature}'.")
                    reference_feature_invalid = True
                    break

                if reference_feature in feature_list:
                    feature_index = feature_list.index(reference_feature)
                    feature_type = feature_types[feature_index]

                    l_reference_features.append(reference_feature)
                    l_reference_types.append(feature_type)
                elif reference_feature == dependent_variable:
                    l_reference_features.append(dependent_variable)
                    l_reference_types.append(dependent_variable_type)
                elif reference_feature in columns:
                    l_reference_features.append(reference_feature)
                    l_reference_types.append("float") # THIS COULD BE A STRING OR A NUMBER, BUT I SET IT TO FLOAT EITHER WAY BECAUSE STRING SIMILARITY IS NOT IMPLEMENTED
                else:
                    print(f"Error: Feature '{reference_feature}' is not a valid feature. ")
                    reference_feature_invalid = True

            if reference_feature_invalid:
                continue

            l_reference_values = []
            reference_value_invalid = False

            print("Type in the reference values for each respective feature.")
            for reference_feature, reference_type in zip(l_reference_features, l_reference_types):
                while True:
                    reference_value = input(f"{reference_feature} (type: '{reference_type}'): ")

                    if reference_type == 'float' or reference_type == 'int':
                        try:
                            reference_value = float(reference_value)
                            l_reference_values.append(reference_value)
                        except Exception as e:
                            print(f"Error: Given value '{reference_value}' is not a number.")
                            continue
                    elif reference_type == 'str':
                        print(f"Error: Value must be a number. Cannot measure string similarity, only euclidean distance.")
                        l_reference_values.append(reference_value)
                    break

            if reference_value_invalid:
                continue

            reference_features = l_reference_features
            reference_values = l_reference_values

            break
    elif result != 'n':
        print("Error: Invalid input. Must be 'y' or 'n'.")
        continue

    break

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

should_normalize = False
normalization_warned = False
while True:
    if not normalization_warned:
        print("\nNote: Normalizing the dataset for training results in (different) weights trained on the normalized values.")
        print("This means that it loses some interpretation value compared to training on non-normalized data.")
        print("However, it tends to make data processing faster among other things.")
        print("The model should perform similarly given the ideal parameters, however, it might be harder to interpret how the features influence the predicted value solely by looking at the weights with the naked eye.")
        print("i.e. In a model that predicts house prices with respect to its area (m²), the feature weight B1 which represents the area would be equal to the predicted price per square meter of the entire dataset.")
        print("You wouldn't be able to infer this from the normalized weights.")
        print("Restricts training data to [0, 1]")
        normalization_warned = True

    
    should_normalize = input("Normalize dataset? (y/n) ")

    if should_normalize == 'y':
        should_normalize = True
    elif should_normalize == 'n':
        should_normalize = False
    else:
        print("Invalid input. Should be 'y' or 'n'.")
        continue
    
    break


TOTAL_STEPS = 50
while True:
    given_steps = input("How many training steps? (must be a number) ")
    try:
        TOTAL_STEPS = int(given_steps)
    except:
        print("Error: Could not convert given training steps to an integer.")
        continue

    break

user_weights = []
use_predefined_weights = False
K = len(feature_list)
while True:
    result = input("Use random weights? (Otherwise you will have to type your own) (y/n) ")
    if result == 'n':
        print("Note: Weight must be a number.")
        use_predefined_weights = True

        for idx in range(K + 1):
            while True:
                if idx != 0:
                    value = input(f"B{idx} ({feature_list[idx - 1]}): ")
                else:
                    value = input("B0: ")

                try:
                    weight = float(value)
                    user_weights.append(weight)
                except Exception as e:
                    print(f"Error: Given value {value} is not a number.")
                    continue

                break

    elif result != 'y':
        print("Error: Invalid input. Must be 'y' or 'n'.")
        continue

    break




feature_min_maxes = []
for feature_type in feature_types:
    if feature_type == "str":
        feature_min_maxes.append([0, -math.inf])
    elif feature_type == "float" or feature_type == "int":
        feature_min_maxes.append([math.inf, -math.inf])

for entry in filtered_entries:
    for feature, idx in zip(feature_list, range(K)):
        feature_min_maxes[idx][0] = min(entry.features[feature], feature_min_maxes[idx][0])
        feature_min_maxes[idx][1] = max(entry.features[feature], feature_min_maxes[idx][1])

N = len(filtered_entries)

if should_normalize:
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

# Matrix as NDarray
B = np.zeros((K + 1, 1), dtype=np.float64)

# Matrix as Matrix
B = np.matrix(B)

if not use_predefined_weights:
    for row in range(K + 1):
        B[row, 0] = random.randint(1, 10)
else:
    for row in range(K + 1):
        B[row, 0] = user_weights[row]

# Partials matrix
# Matrix as NDarray
P = np.zeros((K + 1, 1), dtype=np.float64)

# Matrix as Matrix
P = np.matrix(P)
for row in range(K + 1):
    P[row, 0] = random.randint(1, 10)

LR = 0.5
while True:
    given_lr = input("Learning rate (must be a number): ")
    try:
        LR = float(given_lr)
    except:
        print("Error: Could not convert given learning rate to a float.")
        continue

    break

parameters = [B]
# iterations = [iteration + 1 for iteration in range(TOTAL_STEPS)]
# loss = [0 for _ in range(TOTAL_STEPS)]
iterations = []
loss = []

def animate(i):
    global predicted_line
    
    try:
        if len(feature_list) == 1:
            plt.subplot(1, 2, 2)

        plt.cla()
        try:
            plt.plot(iterations, loss)
        except:
            # Throws exception from time to time as `iterations` and `loss` have unmatching dimensions, as matplotlib calls this function `animate`
            # from another thread, it results in a race condition
            return

        legends = []
        for row in range(0, K + 1):
            if row == 0:
                legend = Line2D([0], [0], label=f"B0: {B[row, 0]} | P0: {P[row, 0]}")
            else:
                legend = Line2D([0], [0], label=f"B{row} ({feature_list[row - 1]}): {B[row, 0]} | P{row}: {P[row, 0]}")
            legends.append(legend)

        plt.legend(handles=legends)

        if len(feature_list) == 1:
            plt.subplot(1, 2, 1)

            predicted_line.remove()
            y = B[0, 0] + (B[1, 0] * x)

            predicted_line, = plt.plot(x, y, c='r')

    except Exception as e:
        print(e)
        pass
    
    if finished:
        ani.event_source.stop()
        print("Stopped animating the graph. Close it in order to predict values.")

finished = False
update_interval = 250
while True:
    update_interval = input("How often should the loss graph update (interval in milliseconds)? ")
    try:
        update_interval = float(update_interval)
    except:
        print("Error: Invalid interval value. Should be a number.")
        continue

    break

training_thread = threading.Thread(target=train, args=(filtered_entries, TOTAL_STEPS))
training_thread.start()

# figure, axes = plt.subplots()
# figure.tight_layout()

if len(feature_list) == 1:
    plt.subplot(1, 2, 1)
    plt.title("Predicted line")
    plt.xlabel(feature_list[0])
    plt.ylabel(dependent_variable)

    # Draw scatter plot of all entries and create the linear space for the input
    point_xs = []
    point_ys = []
    colors = []
    for entry in filtered_entries:
        point_x = entry.features[feature_list[0]]
        point_y = entry.dependent_variable_value

        point_xs.append(point_x)
        point_ys.append(point_y)

        if reference_features != []:
            color = [entry.distance / max_distance, 1 - (entry.distance / max_distance), 0]
            colors.append(color)
        else:
            color = [0, 1, 0]
            colors.append(color)

    
    plt.scatter(point_xs, point_ys, c=colors)

    if should_normalize:
        x = np.linspace(0, 1)
    else:
        x = np.linspace(feature_min_maxes[0][0], feature_min_maxes[0][1])

    y = B[0, 0] + (B[1, 0] * x)
    predicted_line, = plt.plot(x, y, c='r')

    plt.subplot(1, 2, 2)
    plt.title("Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 1)
else:
    plt.title("Loss")
    plt.title("Loss")
    plt.xlabel("Iterations")

ani = FuncAnimation(plt.gcf(), animate, interval=update_interval)
plt.show()

training_thread.join()
predict()

