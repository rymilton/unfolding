import data_handler as dh
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

def reweight(events, classifier):
    class_probabilities = classifier.predict_proba(events)
    data_probability = class_probabilities[:,1]
    weights = data_probability / (1. - data_probability)
    return np.squeeze(np.nan_to_num(weights))

def omnifold(MC_entries, sim_entries, measured_entries, pass_reco_mask, pass_truth_mask, num_iterations, params, train_test_split = False, MC_weight = None, sim_weight = None , data_weight = None):
    verbose = False
    
    sim_entries = sim_entries[pass_truth_mask]
    MC_entries = MC_entries[pass_truth_mask]
    pass_reco_mask = pass_reco_mask[pass_truth_mask]
    if MC_weight is None:
        MC_weight = np.ones(len(MC_entries))
    if sim_weight is None:
        sim_weight = np.ones(len(sim_entries))
    if data_weight is None:
        data_weight = np.ones(len(measured_entries))

    if train_test_split:
        MC_train, MC_test, sim_train, sim_test, pass_reco_train, pass_reco_test, MC_weight_train, _, sim_weight_train, _= train_test_split(MC_entries,
                                                                                                                                           sim_entries,
                                                                                                                                           pass_reco_mask,
                                                                                                                                           MC_weight,
                                                                                                                                           sim_weight,
                                                                                                                                           test_size = .5)
        weights_pull_test = np.ones(len(MC_test))
        weights_push_test = np.ones(len(MC_test))
        weights_test = np.empty(shape=(num_iterations, 2, len(MC_test)))
    else:
        MC_train = MC_entries
        sim_train = sim_entries
        pass_reco_train = pass_reco_mask
        sim_weight_train = sim_weight
        MC_weight_train = MC_weight
    
    print(sim_weight_train[pass_reco_train])
    measured_labels = np.ones(len(measured_entries))
    MC_labels = np.zeros(len(MC_train))

    weights_pull_train = np.ones(len(MC_train))
    weights_push_train = np.ones(len(MC_train))
    weights_train = np.empty(shape=(num_iterations, 2, len(MC_train)))

    step1_classifier = GradientBoostingClassifier(**params)
    step2_classifier = GradientBoostingClassifier(**params)
    for i in range(num_iterations):
        print(f"Starting iteration {i}") 
        step1_data = np.concatenate((sim_train[pass_reco_train], measured_entries))
        step1_labels = np.concatenate((np.zeros(len(sim_train[pass_reco_train])), np.ones(len(measured_labels))))
        step1_weights = np.concatenate((weights_push_train[pass_reco_train]*sim_weight_train[pass_reco_train], data_weight*np.ones(len(measured_entries))))
        
        # Training step 1 classifier and getting weights
        step1_classifier.fit(step1_data, step1_labels, sample_weight = step1_weights)
        new_weights_train = np.ones_like(weights_pull_train)
        new_weights_train[pass_reco_train] = reweight(sim_train[pass_reco_train], step1_classifier)
        
        # Training a regression model to predict the weights of the events that don't pass reconstruction
        if len(sim_train[~pass_reco_train]) > 0:
            regressor_train = GradientBoostingRegressor(**params)
            regressor_train.fit(MC_train[pass_reco_train], new_weights_train[pass_reco_train])
            new_weights_train[~pass_reco_train] = regressor_train.predict(MC_train[~pass_reco_train])
        weights_pull_train = np.multiply(weights_push_train, new_weights_train)

        # Repeating on the test data
        if train_test_split:
            new_weights_test = np.ones_like(weights_pull_test)
            new_weights_test[pass_reco_test] = reweight(sim_test[pass_reco_test], step1_classifier)
            if len(sim_test[~pass_reco_test]) > 0:
                regressor_test = GradientBoostingRegressor(**params)
                regressor_test.fit(MC_test[pass_reco_test], new_weights_test[pass_reco_test])
                new_weights_test[~pass_reco_test] = regressor_test.predict(MC_test[~pass_reco_test])
            weights_pull_test = np.multiply(weights_push_test, new_weights_test)
            
            # Testing step 1 classifier
            if verbose:
                step1_test_accuracy = step1_classifier.score(sim_test[pass_reco_test], np.zeros(len(sim_test[pass_reco_test])))
                print(f"Iteration {i+1}, Step 1 Test Accuracy: {step1_test_accuracy}")        

        # Training step 2 classifier
        step2_data = np.concatenate((MC_train, MC_train))
        step2_labels = np.concatenate((MC_labels, np.ones(len(MC_train))))
        step2_weights = np.concatenate((np.ones(len(MC_train))*MC_weight_train, weights_pull_train*MC_weight_train))
        step2_classifier.fit(step2_data, step2_labels, sample_weight = step2_weights)
        
        # Testing step 2 classifier
        if train_test_split:
            if verbose:
                step2_train_accuracy = step2_classifier.score(step2_data, step2_labels, step2_weights)
                print(f"Iteration {i+1}, Step 2 Train Accuracy: {step2_train_accuracy}")
                step2_test_accuracy = step2_classifier.score(MC_test, np.zeros(len(MC_test)))
                print(f"Iteration {i+1}, Step 2 Test Accuracy: {step2_test_accuracy}")

        # Getting step 2 weights and storing iteration weights
        weights_push_train = reweight(MC_train, step2_classifier)
        weights_train[i, 0], weights_train[i, 1] = weights_pull_train, weights_push_train
        if train_test_split:
            weights_push_test = reweight(MC_test, step2_classifier)
            weights_test[i, 0], weights_test[i, 1] = weights_pull_test, weights_push_test
    if train_test_split:
        return weights_test, MC_test, sim_test[pass_reco_test], pass_reco_test
    else:
        return weights_train

def binned_omnifold(response, measured_hist, num_iterations, params=None):
    if params is None:
        params=dict()
    measured_counts, measured_bin_centers = dh.TH1_to_numpy(measured_hist)
    response_hist = response.HresponseNoOverflow()
    response_counts, response_bin_centers = dh.TH2_to_numpy(response_hist)
    MC_entries, sim_entries = dh.prepare_response_data(response_counts.flatten(), response_bin_centers.flatten())
    measured_entries = dh.prepare_hist_data(measured_counts, measured_bin_centers)
    pass_reco_mask = np.full_like(np.ones(len(MC_entries)), True, dtype=bool)
    pass_truth_mask = np.full_like(np.ones(len(sim_entries)), True, dtype=bool)
    return omnifold(MC_entries, sim_entries, measured_entries, pass_reco_mask, pass_truth_mask, num_iterations, params)
def unbinned_omnifold(MC_entries, sim_entries, measured_entries, pass_reco_mask, pass_truth_mask, num_iterations, train_test_split = False, MC_weight = None, sim_weight = None, data_weight = None, params=None):
    if params is None:
        params = dict()
    if MC_entries.shape[-1] == len(MC_entries):
        MC_entries = np.expand_dims(MC_entries, axis = 1)
    if sim_entries.shape[-1] == len(sim_entries):
        sim_entries = np.expand_dims(sim_entries, axis = 1)
    if measured_entries.shape[-1] == len(measured_entries):
        measured_entries = np.expand_dims(measured_entries, axis = 1)
    return omnifold(MC_entries, sim_entries, measured_entries, pass_reco_mask, pass_truth_mask, num_iterations, params, train_test_split, MC_weight, sim_weight, data_weight)