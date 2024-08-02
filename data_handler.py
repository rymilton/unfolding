import ROOT
import numpy as np

def smear(xt):
  xeff = 0.3 + (1.0-0.3)/20*(xt+10.0)  #  efficiency                                                                                  
  x = ROOT.gRandom.Rndm()
  if x>xeff: return None
  xsmear = ROOT.gRandom.Gaus(-2.5,0.2)     #  bias and smear 
  return xt + xsmear

def CaloSmear(xt, xmin, xmax, a, b, c):
    # calculate efficiency
    diff = xmax - xmin
    center = xmin + diff/2
    xeff = 0.1 + 0.7 * (1 - ((center - xt)/diff)**2 )

    x = ROOT.gRandom.Rndm()
    if x>xeff: 
        return None
    
    sigma_x = xt * (a/ROOT.TMath.Sqrt(xt)) + b
    xsmear = ROOT.gRandom.Gaus(c, sigma_x)

    return xt + xsmear

def smearing_wrapper(xt, smearing_function):
    if smearing_function == "calo":
        return CaloSmear(xt, 0, 10, .15, .1, -1.25)
    else:
        return smear(xt)

def TH1_to_numpy(hist):
    num_bins = hist.GetNbinsX()
    hist_counts = np.empty(num_bins)
    bin_centers = np.empty(num_bins)
    for i in range(num_bins):
        hist_counts[i] = hist.GetBinContent(i+1)
        bin_centers[i] = hist.GetBinCenter(i+1)
    return hist_counts, bin_centers
def TH2_to_numpy(hist):
    num_X_bins = hist.GetNbinsX()
    num_Y_bins = hist.GetNbinsY()
    hist_counts = np.empty(shape=(num_X_bins, num_Y_bins))
    bin_centers = np.empty(shape=(num_X_bins, num_Y_bins), dtype=object)
    for i in range(num_X_bins):
        for j in range(num_Y_bins):
            hist_counts[i, j] = hist.GetBinContent(i+1, j+1)
            bin_center_tuple = (hist.GetXaxis().GetBinCenter(i+1), hist.GetYaxis().GetBinCenter(j+1))
            bin_centers[i, j] = bin_center_tuple
    return hist_counts, bin_centers

def prepare_hist_data(counts, bin_centers):
    out_array = np.empty(shape=(int(np.sum(counts)), 1))
    entry_tracker = 0
    for (count, bin_center) in zip(counts, bin_centers):
        out_array[entry_tracker:int(entry_tracker+count)] = bin_center
        entry_tracker += int(count)
    return out_array

def prepare_response_data(counts, bin_centers):
    truth_array = np.empty(shape=(int(np.sum(counts)), 1))
    sim_array = np.empty(shape=(int(np.sum(counts)), 1))
    entry_tracker = 0
    for (count, bin_center) in zip(counts, bin_centers):
        sim_array[entry_tracker:int(entry_tracker+count)] = bin_center[0]
        truth_array[entry_tracker:int(entry_tracker+count)] = bin_center[1]
        entry_tracker += int(count)
    return truth_array, sim_array
                
