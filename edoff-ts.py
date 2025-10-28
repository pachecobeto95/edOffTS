import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import softmax
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.stats import beta, rankdata

data   = pd.read_csv("update_ee_data_branches.csv")
d_data = pd.read_csv("ee_data_branches.csv") 

length     = len(data)                      # total amount of measurements
n_branches = 3                              # no of exit branches 
n_classes  = 10                             # no of classes to be classified
gamma      = 0.3                            # threshold
neutral    = np.full(n_branches, 1)         # temperature vector initializer
n_tests    = 20                             # no of temperatures derived for a given choice of beta 
beta       = np.linspace(0.0,1.0,n_tests)   # distribution of beta values in edoff-ts

correct      = np.zeros((n_branches,length))
delta        = np.zeros((n_branches,length))

correct[1]   = data["correct_branch_1"].values
correct[2]   = data["correct_branch_2"].values

delta[1]     = d_data["delta_inf_time_branch_1"].values
delta[2]     = d_data["delta_inf_time_branch_2"].values

cols  = [f'logit_class_{c}_branch_{b}' for b in range(1, n_branches) for c in range(1, n_classes+1)]
arr   = data[cols].values
logit = arr.reshape(n_branches-1, n_classes, length) # logit[measure][branch][class]

def plot_scatter(x, y, lbl_scatter=' ', lbl_plot=' ', lbl_x=' ', lbl_y=' ', title=' '):
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y,  label=lbl_scatter, alpha=0.1)
    plt.legend()
    plt.xlabel(lbl_x)
    plt.ylabel(lbl_y)
    plt.title(title)
    plt.show() 

def plot_hist(func, data=np.full(1,1)):
    x = np.linspace(0.0,1.0,length)
    plt.figure(figsize=(8,5))
    plt.hist(data, bins=100, density=True, alpha=0.6, label='Adjusted data')
    plt.plot(x, func(x), 'r-', lw=2, label=' ')
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.title(' ')
    plt.legend()
    plt.show()

def calibrated_confidence(l=1, T=neutral): 
    dist       = softmax(logit[l] / T[l], axis=0)
    calib_conf = np.max(dist, axis=0)
    calib_conf = np.nan_to_num(calib_conf, nan=0.0, posinf=1.0, neginf=0.0)
    return calib_conf

def prob_correct_conf(f, l = 1, T = neutral):      # probability of correct prediction and f is calibrated confidence
    cc               = calibrated_confidence(l, T)
    mask             = correct[l+1] == 1 
    samples          = cc[mask]
    kde              = gaussian_kde(samples)
    return kde(f)

def P_l(f, l = 1, T = neutral):                     # expected probability of correct prediction  
                                                    # at l-th exit given calibrated confidence f
                                                    # we have E(c_l | f_l^(T_l)) = p(c_l = 1 | f_l^(T_l) = f) / p(f_l^(T_l) = f)
                                                    # we calculate it by estimating p(c_l = 1 | f_l^(T_l) = f) by  
                                                    # means of p(f_l^(T_l) = f | c_l = 1)p(c_l =  1) which is equal to the original
                                                    # expression by bayes' theorem
    cc = calibrated_confidence(l, T)
    cc = np.nan_to_num(cc, nan=0.0, posinf=1.0, neginf=0.0)

    mask = correct[l+1] == 1
    conf_correct = cc[mask]

    if len(np.unique(cc)) < 2 or len(np.unique(conf_correct)) < 2:
        base_correct = np.mean(correct[l+1])
        return base_correct

    kde_all = gaussian_kde(cc)
    kde_corr = gaussian_kde(conf_correct)

    p_conf = max(kde_all(f), 1e-12)
    p_conf_corr = kde_corr(f)

    base_correct = np.mean(correct[l+1])

    result = (p_conf_corr * base_correct) / p_conf
    return np.clip(result, 0.0, 1.0)


def capital_p(T=neutral):                                 # proportion of correctly classified 
    classified = 0                                        # over total samples
    total      = 0
    for l in range(n_branches-1):
        conf_curr   = calibrated_confidence(l, T)
        mask        = conf_curr >= gamma
        total      += np.sum(conf_curr)
        classified += np.sum(conf_curr[mask])
        
    probability = classified / total 
    return probability

def p(arg=0, l=1, T=neutral):
    if l <= 0 or T[l] == 0 or T[l-1] == 0:
        return 0.0

    conf_curr = calibrated_confidence(l, T)
    conf_prev = calibrated_confidence(l-1, T)

    mask = conf_prev < gamma
    conf_curr_subset = conf_curr[mask]

    conf_curr_subset = np.nan_to_num(conf_curr_subset, nan=0.0, posinf=1.0, neginf=0.0)

    if len(conf_curr_subset) < 2:
        return 0.0
    if np.std(conf_curr_subset) < 1e-9:
        return 0.0

    kde = gaussian_kde(conf_curr_subset)

    return float(np.clip(kde(arg), 0.0, 1e6))


def Prob_early_exit(T=neutral, l=2):                                # probability of early exit, that is, 
    if l > 0 and T[l] != 0 and T[l-1] != 0:                         # P(f_{l-1}/T_{l-1} < gamma, f_l/T_l >= gamma)
        conf_prev   = calibrated_confidence(l-1, T)                          
        conf_curr   = calibrated_confidence(l, T)
        mask        = (conf_prev < gamma) &  (conf_curr >= gamma)   # obtains cases satisfied by probability condition
        bound       = sum(mask)
        probability = bound / length
        return probability
    return 0

def Acc(T=neutral):
    P = [1.0]*(n_branches-1)
    for l in range(n_branches-1):
        def func_prod(x):                  
            return p(x,l,T)*P_l(x,l,T)
        P[l], error = integrate.quad(func_prod, gamma, 1.0)
        
    accuracy = sum(P) / capital_p(T)
    return accuracy

def Inf_time(T=neutral):                         # Inference time given temperature T 
    P        = np.array([Prob_early_exit(T, l)*delta[l] for l in range(n_branches-1)])
    inf_time = sum(P)
    mean     = inf_time.mean()
    return mean

# Early commentaries by R. Silva.
def edoffts(acc,                # returns a vector of temperatures for the calibrated EE-DNN.
            inf_time,           # function (hyperparameters should be tuned via validation set)
            rounds    = 100,    # function (drawn from observations on validation set)
            beta      = 0.5,    # according to eq (14)
            ln_rate   = 0.1,    # high values may miss good local optima depending on the search space topology
            stable_ln = 1,      # affects convergence velocity for good solutions
            c         = 1,      # exploration intnsity - high values make a difference on learning speed at the beginning
            l         = 1):     # exploration intensity decay - high values may miss good local optima depending on the search space
    
    T_star = np.full(n_branches-1, 1) 
    T_n    = T_star                                                  
    z_T    = (1 - beta)*inf_time(T_n) - beta*acc(T_n)                # by author -- initializing temporary solution
    for n in range(1,rounds+1):
        step     = c/(n**l)                                          # will be used in the finite differences
        phi      = np.random.choice([-1,1], n_branches-1);                  # drawn from a bernoulli distribution and adapted to {-1, 1}
        T_plus   = T_star + step*phi                                 # by author -- increasing finite difference tempereature increment
        T_mins   = T_star - step*phi                                 # by author -- decreasing fininte difference temperature increment
        z_plus   = (1 - beta)*inf_time(T_plus) - beta*acc(T_plus)    # by author -- upper solution 
        z_mins   = (1 - beta)*inf_time(T_mins) - beta*acc(T_mins)    # by author -- lower solution
        phi_m    = [1/phi[k] for k in range(n_branches-1)]                  
        fin_diff = (z_plus - z_mins)/(2*step)
        grad_n   = [x * fin_diff for x in phi_m]
        learn    = ln_rate/(n+stable_ln)
        g_step   = [x * learn for x in grad_n]
        T_n      = np.maximum(T_star - g_step, 1e-6)
        z_n      = (1 - beta)*inf_time(T_n) - beta*acc(T_n)
        if z_n < z_T and min(T_star) > 0: 
            z_T    = z_n
            T_star = T_n
            
    return T_star

temperatures = [edoffts(Acc, Inf_time, beta=beta[attempt]) for attempt in range(n_tests)]

plot_scatter(Inf_time(temperatures), Acc(temperatures), ' ', ' ', r"$E[\Delta | T]$", r"$Acc_{device}$", ' ')



