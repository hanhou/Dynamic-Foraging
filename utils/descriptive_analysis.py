'''
Descriptive analysis for the foraing task

1. Win-stay-lose-shift probabilities
2. Logistic regression on choice and reward history
   Use the model in Hattori 2019 https://www.sciencedirect.com/science/article/pii/S0092867419304465?via%3Dihub
            logit (p_R) ~ Rewarded choice + Unrewarded choice + Choice + Bias
  
Assumed format:
    choice = np.array([0, 1, 1, 0, ...])  # 0 = L, 1 = R
    reward = np.array([0, 0, 0, 1, ...])  # 0 = Unrew, 1 = Reward

Han Hou, Feb 2023
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


def win_stay_lose_shift(choice, reward):
    '''
    Compute p(stay|win), p(shift|lose), and separately for two sides, i.e., p(stay|win in R), etc.
    
    choice = np.array([0, 1, 1, 0, ...])  # 0 = L, 1 = R
    reward = np.array([0, 0, 0, 1, ...])  # 0 = Unrew, 1 = Reward
    
    ---
    return: dict{'p_stay_win', 'p_stay_win_CI', ...}
    '''
    
    stays = np.diff(choice) == 0
    switches = np.diff(choice) != 0
    wins = reward[:-1] == 1
    loses = reward[:-1] == 0
    Ls = choice[:-1] == 0
    Rs = choice[:-1] == 1
    
    p_wsls = {}
    p_lookup = {'p_stay_win':    (stays & wins, wins),   # 'p(y|x)': (y * x, x)
                'p_stay_win_L':  (stays & wins & Ls, wins & Ls),
                'p_stay_win_R':  (stays & wins & Rs, wins & Rs),
                'p_switch_lose': (switches & loses, loses),
                'p_switch_lose_L': (switches & loses & Ls, loses & Ls),
                'p_switch_lose_R': (switches & loses & Rs, loses & Rs),
                }

    for name, (k, n) in p_lookup.items():
        p_wsls[name], p_wsls[name + '_CI'] = _binomial(np.sum(k), np.sum(n))
        
    return p_wsls


def _binomial(k, n):
    '''
    Get p and its confidence interval
    '''
    p = k / n
    return p, 1.96 * np.sqrt(p * (1 - p) / n)


def prepare_logistic(choice, reward, trials_back=15):
    '''    
    Assuming format:
    choice = np.array([0, 1, 1, 0, ...])  # 0 = L, 1 = R
    reward = np.array([0, 0, 0, 1, ...])  # 0 = Unrew, 1 = Reward
    ---
    return: data, Y
    '''
    n_trials = len(choice)
    trials_back = 20
    data = []

    # Encoding data
    RewC, UnrC, C = np.zeros(n_trials), np.zeros(n_trials), np.zeros(n_trials)
    RewC[(choice == 0) & (reward == 1)] = -1   # L rew = -1, R rew = 1, others = 0
    RewC[(choice == 1) & (reward == 1)] = 1
    UnrC[(choice == 0) & (reward == 0)] = -1    # L unrew = -1, R unrew = 1, others = 0
    UnrC[(choice == 1) & (reward == 0)] = 1
    C[choice == 0] = -1
    C[choice == 1] = 1

    for trial in range(trials_back, n_trials):
        data.append(np.hstack([RewC[trial - trials_back : trial],
                            UnrC[trial - trials_back : trial], 
                            C[trial - trials_back : trial]]))
    data = np.array(data)
    Y = C[trials_back:]  # Use -1/1 or 0/1?
    
    return data, Y


def logistic_regression(data, Y, solver='liblinear', penalty='l2', C=1, test_size=0.10):
    '''
    Run one logistic regression fit
    (Reward trials + Unreward trials + Choice + bias)
    Han 20230208
    '''
    trials_back = int(data.shape[1] / 3)
    
    # Do training
    # x_train, x_test, y_train, y_test = train_test_split(data, Y, test_size=test_size)
    logistic_reg = LogisticRegression(solver=solver, fit_intercept=True, penalty=penalty, C=C, n_jobs=1)
    logistic_reg.fit(data, Y)
    output = np.concatenate([logistic_reg.coef_[0], logistic_reg.intercept_])
    
    (logistic_reg.b_RewC, 
    logistic_reg.b_UnrC, 
    logistic_reg.b_C, 
    logistic_reg.bias) = decode_betas(output)
    
    return output, logistic_reg


def logistic_regression_CV(data, Y, Cs=10, cv=10, solver='liblinear', penalty='l2', n_jobs=-1):
    '''
    logistic regression with cross validation
    1. Use cv-fold cross validation to determine best penalty C
    2. Using the best C, refit the model with cv-fold again
    3. Report the mean and CI (1.96 * std) of fitted parameters in logistic_reg_refit
    
    Cs: number of Cs to grid search
    cv: number of folds
    
    -----
    return: logistic_reg_cv, logistic_reg_refit
    
    Han 20230208
    '''
    # Do cross validation, try different Cs
    logistic_reg_cv = LogisticRegressionCV(solver=solver, fit_intercept=True, penalty=penalty, Cs=Cs, cv=cv, n_jobs=n_jobs)
    logistic_reg_cv.fit(data, Y)

    return logistic_reg_cv


def bootstrap(func, data, Y, n_bootstrap=1000, **kwargs):
    # Generate bootstrap samples
    indices = np.random.choice(range(Y.shape[0]), size=(n_bootstrap, Y.shape[0]), replace=True)
    bootstrap_Y = [Y[index] for index in indices]
    bootstrap_data = [data[index, :] for index in indices]
    
    # Fit the logistic regression model to each bootstrap sample
    outputs = np.array([func(data, Y, **kwargs)[0] for data, Y in zip(bootstrap_data, bootstrap_Y)])
    
    # Get bootstrap mean, std, and CI
    bs = {'raw': outputs,
          'mean': np.mean(outputs, axis=0),
          'std': np.std(outputs, axis=0),
          'CI_lower': np.percentile(outputs, 2.5, axis=0),
          'CI_upper': np.percentile(outputs, 97.5, axis=0)}
    
    return bs
    
    
def decode_betas(coef):
    # Decode fitted betas
    coef = np.atleast_2d(coef)
    trials_back = int((coef.shape[1] - 1) / 3)  # Hard-coded
    b_RewC = coef[:, trials_back - 1::-1]
    b_UnrC = coef[:, 2 * trials_back - 1: trials_back - 1:-1]
    b_C = coef[:, 3 * trials_back - 1:2 * trials_back - 1:-1]
    bias = coef[:, -1:]
    return b_RewC, b_UnrC, b_C, bias


def logistic_regression_bootstrap(data, Y, n_bootstrap=1000, **kwargs):
    '''
    1. use cross-validataion to determine the best L2 penality parameter, C
    2. use bootstrap to determine the CI and std
    '''
    
    # Cross validation
    logistic_reg = logistic_regression_CV(data, Y, **kwargs)
    best_C = logistic_reg.C_
    para_mean = np.hstack([logistic_reg.coef_[0], logistic_reg.intercept_])
    
    (logistic_reg.b_RewC, 
     logistic_reg.b_UnrC, 
     logistic_reg.b_C, 
     logistic_reg.bias) = decode_betas(para_mean)
    
    # Bootstrap
    if n_bootstrap > 0:
        bs = bootstrap(logistic_regression, data, Y, n_bootstrap=n_bootstrap, C=best_C[0], **kwargs)
        
        logistic_reg.coefs_bootstrap = bs
        (logistic_reg.b_RewC_CI, 
        logistic_reg.b_UnrC_CI, 
        logistic_reg.b_C_CI, 
        logistic_reg.bias_CI) = decode_betas(np.vstack([bs['CI_lower'], bs['CI_upper']]))

        # # Override with bootstrap mean
        # (logistic_reg.b_RewC, 
        # logistic_reg.b_UnrC, 
        # logistic_reg.b_C, 
        # logistic_reg.bias) = decode_betas(np.vstack([bs['mean'], bs['mean']]))
    
    return logistic_reg
    

# ----- Plotting functions -----
            
def plot_logistic_regression(logistic_reg, ax=None, ls='-o'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        
    # return 
    if_CV = hasattr(logistic_reg, 'b_RewC_CI') # If cross-validated
    x = np.arange(1, logistic_reg.b_RewC.shape[1] + 1)
    plot_spec = {'b_RewC': 'g', 'b_UnrC': 'r', 'b_C': 'b', 'bias': 'k'}    

    for name, col in plot_spec.items():
        mean = getattr(logistic_reg, name)
        ax.plot(x if name != 'bias' else 1, np.atleast_2d(mean)[0, :], ls + col, label=name + ' $\pm$ CI')

        if if_CV:  # From cross validation
            CI = np.atleast_2d(getattr(logistic_reg, name + '_CI'))
            ax.fill_between(x=x if name != 'bias' else [1], 
                            y1=CI[0, :], 
                            y2=CI[1, :],
                            color=col,
                            alpha=0.3)
        
    if if_CV and hasattr(logistic_reg, "scores_"):
        score_mean = np.mean(logistic_reg.scores_[1.0])
        score_std = np.std(logistic_reg.scores_[1.0])
        if hasattr(logistic_reg, 'cv'):
            ax.set(title=f'{logistic_reg.cv}-fold CV, score $\pm$ std = {score_mean:.3g} $\pm$ {score_std:.2g}\n'
                    f'best C = {logistic_reg.C_[0]:.3g}')
    else:
        pass
        # ax.set(title=f'train: {logistic_reg.train_score:.3g}, test: {logistic_reg.test_score:.3g}')
    
    ax.legend()
    ax.set(xlabel='Past trials', xticks=x, ylabel='Logistic regression coeffs')
    ax.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    
    return ax


def plot_wsls(p_wsls, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    
    x = 0
    xlabel = []
    for i, name in enumerate(['stay_win', 'switch_lose']):    
        ax.bar(x, p_wsls[f'p_{name}'], 
               yerr=p_wsls[f'p_{name}_CI'],
               color='k', label='all')
        x += 1
        for side, col in (('L', 'r'), ('R', 'b')):
            ax.bar(x, p_wsls[f'p_{name}_{side}'], 
                   yerr=p_wsls[f'p_{name}_{side}_CI'],
                   color=col, label=side)
            x += 1
    
    ax.set(xticks=[1, 4], xticklabels=['p(stay | win)', 'p(shift | lose)'])
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:3], l[:3])
    ax.set(ylim=(0, 1))