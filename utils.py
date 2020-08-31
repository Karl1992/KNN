import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    f1 = 2 * sum(np.multiply(np.array(real_labels), np.array(predicted_labels))) / (sum(real_labels) + sum(predicted_labels))
    return f1

class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2, p=3):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :param k: int
        :return: float
        """
        return pow(sum(abs(np.power(list(map(lambda x: x[0] - x[1], zip(point1, point2))), p))), 1/p)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return pow(sum(abs(np.power(list(map(lambda x: x[0] - x[1], zip(point1, point2))), 2))), 1/2)

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return sum(np.multiply(np.array(point1), np.array(point2)))

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        numerator = sum(np.array(point1)*np.array(point2))
        p1_mod = np.sqrt(sum(np.array(point1)**2))
        p2_mod = np.sqrt(sum(np.array(point2)**2))
        if p1_mod == 0 or p2_mod == 0:
            return '?'
        else:
            denominator = p1_mod * p2_mod
            return 1-numerator/denominator

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        return -np.exp(-sum(np.power((np.array(point1)-np.array(point2)), 2))/2)


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = 'euclidean'
        self.best_scaler = 'min_max_scale'
        self.best_model = None
        self.method_priority = {'euclidean': 1, 'minkowski': 2, 'gaussian': 3, 'inner_prod': 4, 'cosine_dist': 5}
        self.scaler_priority = {'min_max_scale':1, 'normalize':2}
        self.result_without_scale = []
        self.result_with_scale = []

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_scaler = None
        f1_best = 0
        for k in range(1, 30, 2):
            for method in distance_funcs.keys():
                tuning_model = KNN(k, distance_funcs[method])
                tuning_model.train(x_train, y_train)
                prediction = tuning_model.predict(x_val)
                f1 = f1_score(y_val, prediction)
                self.result_without_scale.append([k, method, f1])
                print('k =', k, ' method = ', method , ' f1 score =', f1)
                if f1 > f1_best:
                    f1_best = f1
                    self.best_k = k
                    self.best_distance_function = method
                    self.best_model = tuning_model
                elif f1 == f1_best:
                    if self.method_priority[method] < self.method_priority[self.best_distance_function]:
                        self.best_distance_function = method
                        self.best_model = tuning_model

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        result = []
        f1_best = 0
        for scaler in scaling_classes.keys():
            scaler_f = scaling_classes[scaler]()
            x_train, x_val = scaler_f(x_train), scaler_f(x_val)
            for k in range(1, 30, 2):
                for method in distance_funcs.keys():
                    tuning_model = KNN(k, distance_funcs[method])
                    tuning_model.train(x_train, y_train)
                    prediction = tuning_model.predict(x_val)
                    f1 = f1_score(y_val, prediction)
                    self.result_with_scale.append([k, distance_funcs[method], f1])
                    print('scaler =', scaler, ' k =', k, ' method = ', method, ' f1 score =', f1)
                    if f1 > f1_best:
                        f1_best = f1
                        self.best_k = k
                        self.best_scaler = scaler
                        self.best_distance_function = method
                        self.best_model = tuning_model
                    elif f1 == f1_best:
                        if self.scaler_priority[scaler] < self.scaler_priority[self.best_scaler]:
                            self.best_scaler = scaler
                            self.best_distance_function = method
                            self.best_model = tuning_model
                        elif self.scaler_priority[scaler] == self.scaler_priority[self.best_scaler]:
                            if self.method_priority[method] < self.method_priority[self.best_distance_function]:
                                self.best_distance_function = method
                                self.best_model = tuning_model


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        for i in range(len(features)):
            denominator = np.sqrt(sum(np.array(features[i]) ** 2))
            if denominator != 0:
                features[i] = list(np.array(features[i])/denominator)
        return features

class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.max_list = []
        self.min_list = []
        self.state = {'first_call': True}
        
    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        if self.state['first_call']:
            self.state['first_call'] = False
            storage = [[] for _ in range(len(features[1]))]
            for i in range(len(features)):
                for j in range(len(features[i])):              
                    storage[j].append(features[i][j])
            for i in range(len(storage)):
                self.max_list.append(max(storage[i]))
                self.min_list.append(min(storage[i]))
            for i in range(len(features)):
                features[i] = list((np.array(features[i]) - np.array(self.min_list)) / (np.array(self.max_list) - np.array(self.min_list)))
            return features
        else:
            for i in range(len(features)):
                features[i] = list((np.array(features[i]) - np.array(self.min_list)) / (np.array(self.max_list) - np.array(self.min_list)))
            return features
            
