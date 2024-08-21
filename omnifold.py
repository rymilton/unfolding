import data_handler as dh
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def reweight(events, classifier):
    class_probabilities = classifier.predict_proba(events)
    data_probability = class_probabilities[:,1]
    weights = data_probability / (1. - data_probability)
    return np.squeeze(np.nan_to_num(weights))

def omnifold(MC_entries, sim_entries, measured_entries, num_iterations):

    MC_train, MC_test, sim_train, sim_test = train_test_split(MC_entries, sim_entries, test_size = .5)
    
    sim_labels = np.zeros(len(sim_train))
    measured_labels = np.ones(len(measured_entries))
    MC_labels = np.zeros(len(MC_train))

    step1_data = np.concatenate((sim_train, measured_entries))
    step1_labels = np.concatenate((sim_labels, measured_labels))

    step2_data = np.concatenate((MC_train, MC_train))
    step2_labels = np.concatenate((MC_labels, np.ones(len(MC_train))))

    weights_pull_train = np.ones(len(sim_train))
    weights_push_train = np.ones(len(sim_train))
    weights_train = np.empty(shape=(num_iterations, 2, len(sim_train)))

    weights_pull_test = np.ones(len(sim_test))
    weights_push_test = np.ones(len(sim_test))
    weights_test = np.empty(shape=(num_iterations, 2, len(sim_test)))
    
    for i in range(num_iterations):
        step1_weights = np.concatenate((weights_push_train, np.ones(len(measured_entries))))
        # Training step 1 classifier
        step1_classifier = GradientBoostingClassifier()
        step1_classifier.fit(step1_data, step1_labels, sample_weight = step1_weights)
        weights_pull_train = np.multiply(weights_push_train, reweight(sim_train, step1_classifier))
        weights_pull_test = np.multiply(weights_push_test, reweight(sim_test, step1_classifier))
        # Testing step 1 classifier
        step1_test_accuracy = step1_classifier.score(sim_test, np.zeros(len(sim_test)))
        print(f"Iteration {i+1}, Step 1 Test Accuracy: {step1_test_accuracy}")        

        # Training step 2 classifier
        step2_weights = np.concatenate((np.ones(len(MC_train)), weights_pull_train))
        step2_classifier = GradientBoostingClassifier()
        step2_classifier.fit(step2_data, step2_labels, sample_weight = step2_weights)
        # Testing step 2 classifier
        step2_test_accuracy = step2_classifier.score(MC_test, np.zeros(len(MC_test)))
        print(f"Iteration {i+1}, Step 2 Test Accuracy: {step2_test_accuracy}")        
        
        weights_push_train = reweight(MC_train, step2_classifier)
        weights_push_test = reweight(MC_test, step2_classifier)
        
        weights_train[i, 0], weights_train[i, 1] = weights_pull_train, weights_push_train
        weights_test[i, 0], weights_test[i, 1] = weights_pull_test, weights_push_test
    return weights_test, MC_test, sim_test

def binned_omnifold(response, measured_hist, num_iterations):
    measured_counts, measured_bin_centers = dh.TH1_to_numpy(measured_hist)
    response_hist = response.HresponseNoOverflow()
    response_counts, response_bin_centers = dh.TH2_to_numpy(response_hist)
    MC_entries, sim_entries = dh.prepare_response_data(response_counts.flatten(), response_bin_centers.flatten())
    measured_entries = dh.prepare_hist_data(measured_counts, measured_bin_centers)
    
    return omnifold(MC_entries, sim_entries, measured_entries, num_iterations)
def unbinned_omnifold(MC_data, sim_data, measured_data, num_iterations):
    miss_mask = (~np.isnan(sim_data)) & (~np.isnan(MC_data))
    
    MC_entries = np.expand_dims(MC_data[miss_mask], axis = 1)
    sim_entries = np.expand_dims(sim_data[miss_mask], axis = 1)
    measured_entries = np.expand_dims(measured_data, axis = 1)
    
    return omnifold(MC_entries, sim_entries, measured_entries, num_iterations)