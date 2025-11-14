import numpy as np
import pandas as pd
import config, os, sys, random
from tqdm import tqdm
import ee_nn_calibration
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def exp_beta_function(temp_list, n_branches, threshold, df_edge, df_cloud, beta, overhead, n_classes):

	inf_time, ee_prob, _ = ee_nn_calibration.compute_inference_time(temp_list, n_branches, threshold, df_edge, df_cloud, overhead, n_classes)
	inf_time = 1000*inf_time
	
	exp_acc, exp_ee_prob = ee_nn_calibration.exp_acc_edge(temp_list, n_branches, threshold, df_edge, df_cloud, overhead, n_classes)


	f = inf_time - beta*exp_acc

	#print("Acc Device: %s"%acc)
	#print("Acc Exp: %s"%(exp_acc), exp_ee_prob)
	#print("Inf Time: %s"%(inf_time), a)
	#print(temp_list)

	return f, inf_time, exp_acc, ee_prob



class TemperatureScaler:
    """
    Adjusts the temperature parameter (T) to calibrate the output of a model.
    T is optimized by minimizing the Negative Log-Likelihood (NLL) on a 
    validation set.
    """
    
    def __init__(self, args, temp_list, df_edge, df_cloud, threshold, overhead, beta):
        """Initializes the Temperature T at 1.0 (no initial effect)."""
        self.args = args
        self.temp_list = temp_list
        self.threshold = threshold
        self.overhead = overhead
        self.n_classes = config.dataset_config[args.dataset_name]["n_classes"]
        self.beta = beta

        random_seed = 42

        np.random.seed(random_seed)
        random.seed(random_seed)

        # --- CODE FOR RANDOM DATA SPLIT (50/50) ---
        
        # 1. Split df_edge: Sample 50% for Validation
        
        # frac=0.5 means sample 50% of the rows. 
        # random_state ensures the split is the same every time the code runs.
        self.df_edge_val = df_edge.sample(frac=0.5, random_state=random_seed).reset_index(drop=True)
        
        # The remaining data goes to the Test set
        # 'drop=True' means drop the rows that were sampled for the validation set
        self.df_edge_test = df_edge.drop(self.df_edge_val.index).reset_index(drop=True)

        # 2. Split df_cloud: Ensure the split is the SAME as the edge data for corresponding samples
        
        # We must use the indices determined by the EDGE split to keep the samples aligned.
        # This assumes df_edge and df_cloud have the same original indices.
        
        # Use the indices that went into df_edge_val to sample df_cloud_val
        val_indices = self.df_edge_val.index
        
        # Since df_edge_val and df_edge_test were created using sample/drop, 
        # we need to re-align the original df_cloud to the original df_edge indices.
        
        # Re-sample df_cloud using the same index mask determined by the df_edge split:
        self.df_cloud_val = df_cloud.loc[self.df_edge_val.index].reset_index(drop=True)
        self.df_cloud_test = df_cloud.loc[self.df_edge_test.index].reset_index(drop=True)

        # WARNING: The simple 'sample' approach above assumes df_edge and df_cloud 
        # have identical original indices (0 to N-1). 
        # For simplicity and robustness, let's use a single index sample on the initial DataFrame
        
        # --- ROBUST RANDOM SPLIT (Revised) ---
        n = len(df_edge)
        val_size = int(n * 0.5)

        # Create a randomized array of indices
        all_indices = np.arange(n)
        np.random.shuffle(all_indices) # Shuffle the indices

        # Split indices
        val_indices = all_indices[:val_size]
        test_indices = all_indices[val_size:]
        
        # Select rows using the randomized indices
        self.df_edge_val = df_edge.iloc[val_indices].reset_index(drop=True)
        self.df_edge_test = df_edge.iloc[test_indices].reset_index(drop=True)

        self.df_cloud_val = df_cloud.iloc[val_indices].reset_index(drop=True)
        self.df_cloud_test = df_cloud.iloc[test_indices].reset_index(drop=True)
        

    def get_logits_labels(self):
        # --- Compute calibrated confidences per branch --
        logits_dict, labels_dict = {}, {}
        for i in range(self.args.n_branches):
            logits = np.stack([self.df_edge_val[f"logit_class_{c+1}_branch_{i+1}"] for c in range(self.n_classes)], axis=1)
            labels = self.df_edge_val['target'].values
            logits_dict[i] = logits
            labels_dict[i] = labels

        return logits_dict, labels_dict


    def NLL_loss(self, T, logits, labels):
        """
        Calculates the Negative Log-Likelihood (NLL) for optimization.

        Parameters:
        T (float): The temperature parameter to be optimized.
        logits (np.array): Model's logits (unnormalized outputs).
        labels (np.array): True class labels.

        Returns:
        float: The NLL value (to be minimized).
        """
        # Ensure T is positive
        if T.any() <= 0:
            return np.inf

        # Scale the logits: logits_scaled = logits / T
        logits_scaled = logits / T
        
        # Calculate log-softmax for numerical stability (logsumexp)
        log_prob = logits_scaled - logsumexp(logits_scaled, axis=1, keepdims=True)
        
        # Get the log probability of the true class for each sample
        correct_indices = labels
        
        # log_likelihood = sum( log(P_correct) )
        log_likelihood = log_prob[np.arange(len(labels)), correct_indices].sum()
        
        # NLL = -log_likelihood / N_samples (Average NLL)
        nll = -log_likelihood / len(labels)
        
        return nll

    def min(self):
        """
        Fits the parameter T by minimizing the NLL.

        Parameters:
        logits (np.array): Model's logits.
        labels (np.array): True class labels.
        """
        print("Starting temperature (T) adjustment...")

        logits_dict, labels_dict = self.get_logits_labels()

        # Optimization bounds (T must be positive, e.g., T >= 0.01)
        bounds = ((0.01, None),) 

        for i in range(self.args.n_branches):
            # Use L-BFGS-B for bounded optimization
            res = minimize(self.NLL_loss, self.temp_list, args=(logits_dict[0], labels_dict[0]), 
                method='L-BFGS-B', 
                bounds=bounds
            )

            self.temp_list[i] = res.x[0]

        final_loss, inf_time, acc, ee_prob = exp_beta_function(self.temp_list, self.args.n_branches, 
            self.threshold, self.df_edge_test, self.df_cloud_test, self.beta, self.overhead, self.n_classes)
        
        print("Optimization completed. Best Loss: %s, Acc Edge: %s, Inf Time: %s, EE Prob: %s"
            %(final_loss, acc, inf_time, ee_prob))

        return self.temp_list, final_loss, inf_time, acc, ee_prob

        #if res.success:
        #    self.temp_list = res.x
        #    print(f"Adjustment finished. Optimal Temperature (T): {self.temp_list:.4f}")
        #    print(f"Minimum NLL achieved: {res.fun:.4f}")
        #else:
        #    print(f"Warning: Optimization did not converge. Reason: {res.message}")
        #    self.T = T_init[0] # Keep T=1.0 if optimization fails

    def scale(self, logits):
        """
        Applies temperature scaling to the logits.

        Parameters:
        logits (np.array): Original logits.

        Returns:
        np.array: Logits scaled by the temperature T.
        """
        if self.T <= 0:
            warnings.warn("Invalid Temperature T (<= 0). Returning original logits.", RuntimeWarning)
            return logits
            
        return logits / self.T