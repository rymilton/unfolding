import ROOT
import numpy as np
import data_handler as dh

def normalize_response(response, MC_hist):
    num_measured_bins = len(response)
    num_MC_bins = len(MC_hist)
    normalized_response = np.empty((num_measured_bins, num_MC_bins))
    for i in range(num_measured_bins):
        for j in range(num_MC_bins):
            if MC_hist[j] > 0:
                normalized_response[i, j] = response[i,j] / MC_hist[j]
    return normalized_response

def manual_IBU(response, MC_hist, measured_data, num_iterations):
    num_MC_bins = len(MC_hist)
    num_measured_bins = len(measured_data)
    
    # Calculate the efficiency
    efficiency = np.zeros(num_MC_bins)
    for i in range(num_MC_bins):
        n_truth_reco = np.sum(response[:, i])
        n_truth = MC_hist[i]
        efficiency[i] = n_truth_reco / n_truth if n_truth > 0 else 0
        
    # Normalizing the response matrix
    response = normalize_response(response, MC_hist)

    # IBU procedure
    t_distributions = []
    for n in range(num_iterations):
        print("Iteration", n)
        t = np.empty(num_MC_bins)
        if n == 0:
            t = MC_hist
        else:
            for j in range(num_MC_bins):
                total_sum = 0
                for i in range(num_measured_bins):
                    numerator = response[i, j] * t_distributions[-1][j] * measured_data[i]
                    denominator = 0
                    for k in range(num_MC_bins):
                        denominator += response[i,k] * (t_distributions[-1][k])
                    if denominator > 0:
                        total_sum += numerator/denominator
                    t[j] = total_sum
        if n > 0:
            t = np.divide(t, efficiency, out=np.zeros_like(t, dtype=np.float32), where=efficiency!=0)
        t_distributions.append(t)
    return t_distributions

def manual_IBU_np(sim_data, MC_data, sim_bins, MC_bins, sim_low, sim_high, MC_low, MC_high, measured_hist, num_iterations):
    # Converting response matrix to a 2D histogram and normalizing it to get probabilities
    measured_data, _ = dh.TH1_to_numpy(measured_hist)

    miss_mask = (~np.isnan(sim_data)) & (~np.isnan(MC_data))
    
    # Calculating the response matrix and effiency
    MC_hist = np.histogram(MC_data[~np.isnan(MC_data)], bins = MC_bins, range = [MC_low, MC_high])[0]
    
    response = np.histogram2d(sim_data[miss_mask], MC_data[miss_mask], bins=[sim_bins, MC_bins], range=[[sim_low,sim_high],[MC_low,MC_high]])[0]
    return manual_IBU(response, MC_hist, measured_data, num_iterations)

def manual_IBU_ROOT(response_matrix, measured_hist, num_iterations):
    
    true_hist = response_matrix.Htruth()
    
    measured_array, _ = dh.TH1_to_numpy(measured_hist)
    MC_array, _ = dh.TH1_to_numpy(true_hist)
    
    unnormalized_response_hist = response_matrix.HresponseNoOverflow()
    num_truth_bins = unnormalized_response_hist.GetNbinsY()
    response_array, _= dh.TH2_to_numpy(unnormalized_response_hist)

    t_distributions = manual_IBU(response_array, MC_array, measured_array, num_iterations)

    unfolded_distribution_hist = ROOT.TH1D("unfolded_distribution_hist",
                                           "unfolded_distribution_hist",
                                           num_truth_bins,
                                           unnormalized_response_hist.GetYaxis().GetBinLowEdge(1),
                                           unnormalized_response_hist.GetYaxis().GetBinLowEdge(num_truth_bins)
                                           +unnormalized_response_hist.GetYaxis().GetBinWidth(num_truth_bins)
                                          )
    for i, value in enumerate(t_distributions[-1]):
        unfolded_distribution_hist.SetBinContent(i+1, value)
    return unfolded_distribution_hist

def no_ml_omnifold(response, MC_hist, measured_data, num_iterations):
    num_MC_bins = len(MC_hist)
    num_measured_bins = len(measured_data)
    
    # Calculate the efficiency
    efficiency = np.zeros(num_MC_bins)
    for i in range(num_MC_bins):
        n_truth_reco = np.sum(response[:, i])
        n_truth = MC_hist[i]
        efficiency[i] = n_truth_reco / n_truth if n_truth > 0 else 0
        
    # Normalizing the response matrix
    response = normalize_response(response, MC_hist)

    # Omnifold procedure
    t_distributions = []
    omega_distributions = []
    nu_distributions = []
    for n in range(num_iterations):
        print("Iteration", n)
        t = np.zeros(num_MC_bins)
        if n == 0:
            t_distributions.append(MC_hist)
        else:
            for j in range(num_MC_bins):
                for i in range(num_measured_bins):
                    t[j] += response[i, j] * t_distributions[-1][j] * omega_distributions[-1][i]
            t = np.divide(t, efficiency, out=np.zeros_like(t, dtype=np.float32), where=efficiency!=0)
            t_distributions.append(t)

        nu = np.divide(t_distributions[-1], t_distributions[0], out=np.zeros_like(t_distributions[-1], dtype = np.float32), where=t_distributions[0]!=0)
        
        omega = np.empty(num_measured_bins)
        for i in range(num_measured_bins):
            denominator = 0
            for j in range(num_MC_bins):
                denominator += (response[i, j] * t_distributions[-1][j])
            if denominator > 0:
                omega[i] = measured_data[i]/denominator
            else:
                omega[i] = 0
        omega_distributions.append(omega)
        nu_distributions.append(nu)

    return nu_distributions, omega_distributions

def no_ml_omnifold_np(sim_data, MC_data, sim_bins, MC_bins, sim_low, sim_high, MC_low, MC_high, measured_hist, num_iterations):
    # Converting response matrix to a 2D histogram and normalizing it to get probabilities
    measured_data, _ = dh.TH1_to_numpy(measured_hist)
    
    miss_mask = (~np.isnan(sim_data)) & (~np.isnan(MC_data))
    
    # Calculating the response matrix and effiency
    MC_hist = np.histogram(MC_data[~np.isnan(MC_data)], bins = MC_bins, range = [MC_low, MC_high])[0]
    response = np.histogram2d(sim_data[miss_mask], MC_data[miss_mask], bins=[sim_bins, MC_bins], range=[[sim_low,sim_high],[MC_low,MC_high]])[0]

    return no_ml_omnifold(response, MC_hist, measured_data, num_iterations)

def no_ml_omnifold_ROOT(response_matrix, measured_hist, num_iterations):
    true_hist = response_matrix.Htruth()
    sim_hist = response_matrix.Hmeasured()
    
    measured_array, _ = dh.TH1_to_numpy(measured_hist)
    MC_array, _ = dh.TH1_to_numpy(true_hist)
    sim_array, _ = dh.TH1_to_numpy(sim_hist)
    
    unnormalized_response_hist = response_matrix.HresponseNoOverflow()
    num_truth_bins = unnormalized_response_hist.GetNbinsY()
    num_measured_bins = unnormalized_response_hist.GetNbinsX()
    response_array, _ = dh.TH2_to_numpy(unnormalized_response_hist)
        
    return no_ml_omnifold(response_array, MC_array, measured_array, num_iterations)