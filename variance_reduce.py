import time
import numpy as np
from tqdm import tqdm
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt


# Black and Scholes model
def d1(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))


def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))


def BlackScholes(S0, K, r, sigma, T):
    return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - \
           K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))


# get correlation:
def corr_c(conts, targs, conts_expt):
    cc = -np.sum((targs - np.mean(targs))*(conts - \
         conts_expt))/(np.sum((conts - conts_expt)**2))
    return cc


def heston_single(mu, rho, kappa, sigmasquare, theta, init_price,
               T, step_n, r, W, Z):
    """ single simulation of Heston model by Milstein scheme
    parameters:
        rho: correlation parameter
        sigmasquare: initial volatility
        init_price: initial price of asset
        T: time to maturity
        step_n: the steps of every simulation
        r: interest rate
        W: first Brownian Motion
        Z: second Brownian Motion
    return:
        the final asset price of this simulation
    """
    delta_t = T/step_n
    x = 0
    for i in range(0, step_n):
        delta_1 = (r-0.5*sigmasquare)*delta_t
        delta_2 = rho*np.sqrt(sigmasquare)*W[i]
        delta_3 = np.sqrt(1-rho**2)*np.sqrt(sigmasquare)*Z[i]
        x = x + delta_1 + delta_2 + delta_3
        # update sigmasquare
        sigmasquare = sigmasquare + kappa*(theta-sigmasquare)*delta_t +\
                      mu*np.sqrt(sigmasquare)*W[i] +\
                      ((mu**2)/4)*(W[i]**2-delta_t)
        if sigmasquare < 0:
            sigmasquare = -sigmasquare
    asset_final = init_price*np.exp(x)
    return asset_final


def heston(mu, rho, kappa, sigmasquare0, theta, init_price,
              sim_n, step_n, T, K, r):
    """ normal Monte Carlo Milstein simulation
    sim_n: the number of simulations
    step_n: the number of steps
    """
    payoff = np.zeros([sim_n])
    std = (T / step_n) ** 0.5
    for i in range(0, sim_n):
        W = np.random.normal(0, std, step_n + 1)
        Z = np.random.normal(0, std, step_n + 1)
        ST = heston_single(mu, rho, kappa, sigmasquare0,
             theta, init_price, T, step_n, r, W, Z)
        payoff[i] = max(ST-K, 0)
    mean = np.mean(payoff)
    variance = np.var(payoff)
    return mean, variance


def correlation(mu, rho, kappa, sigmasquare0, theta,
                    init_price, sim_n, step_n, T, K, r):
    """ get the correlation bwtween control variable and target variable
    sim_n: the number of simulations
    step_n: the number of steps
    """
    payoff = np.zeros([sim_n])
    STs = np.zeros([sim_n])
    std = (T / step_n) ** 0.5
    for i in range(0, sim_n):
        W = np.random.normal(0, std, step_n + 1)
        Z = np.random.normal(0, std, step_n + 1)
        ST = heston_single(mu, rho, kappa, sigmasquare0,
             theta, init_price, T, step_n, r, W, Z)
        STs[i] = ST
        payoff[i] = max(ST-K, 0)
    corre = corr_c(STs, payoff, init_price)
    return corre


def heston_control(mu, rho, kappa, sigmasquare0, theta,
                      init_price, sim_n, step_n, T, K, r, corr):
    """
    Monte Carlo Milstein simulation with control variate
    sim_n: the number of simulations
    step_n: the number of steps
    """
    # estimate correlation first:

    payoff = np.zeros([sim_n])
    std = (T / step_n) ** 0.5
    for i in range(0, sim_n):
        W = np.random.normal(0, std, step_n + 1)
        Z = np.random.normal(0, std, step_n + 1)
        ST = heston_single(mu, rho, kappa, sigmasquare0,
             theta, init_price, T, step_n, r, W, Z)
        payoff[i] = max(ST - K, 0) + corr*(ST - init_price)
    mean = np.mean(payoff)
    variance = np.var(payoff)
    return mean, variance


def heston_anti(mu, rho, kappa, sigmasquare0, theta,
                   init_price, sim_n, step_n, T, K, r):
    """
    Monte Carlo Milstein simulation with antithetic variate
    sim_n: the number of simulations
    step_n: the number of steps
    """
    # get correlation first:
    payoff = np.zeros([sim_n])
    std = (T / step_n) ** 0.5
    for i in range(0, sim_n):
        W = np.random.normal(0, std, step_n + 1)
        Z = np.random.normal(0, std, step_n + 1)
        ST1 = heston_single(mu, rho, kappa, sigmasquare0,
              theta, init_price, T, step_n, r, W, Z)
        payoff1 = max(ST1 - K, 0)
        # antithetic variate method:
        W1 = -W
        Z1 = -Z
        ST2 = heston_single(mu, rho, kappa, sigmasquare0,
              theta, init_price, T, step_n, r, W1, Z1)
        payoff2 = max(ST2 - K, 0)
        payoff[i] = (payoff1 + payoff2)/2
    mean = np.mean(payoff)
    variance = np.var(payoff)
    return mean, variance


def heston_conditional_single(mu, rho, kappa, sigmasquare, theta, init_price,
                            T, step_n, K, W):
    """ single simulation of Heston model by conditional expectation
        Milstein scheme
    parameters:
        rho: correlation parameter
        sigmasquare: initial volatility
        init_price: initial price of asset
        T: time to maturity
        step_n: the steps of every simulation
        W: Brownian Motion
    return:
        the payoff of this simulation
    """
    r = 0  # assume interest is 0
    integral = 0
    integralstoch = 0
    delta_t = T / step_n
    for i in range(0, step_n):
        integral = integral + delta_t * sigmasquare
        integralstoch = integralstoch + (np.sqrt(sigmasquare)) * W[i]
        sigmasquare = sigmasquare + kappa*(theta-sigmasquare)*delta_t +\
                      mu*np.sqrt(sigmasquare)*W[i] +\
                      ((mu**2)/4)*(W[i]**2-delta_t)
        if sigmasquare < 0:
            sigmasquare = - sigmasquare
    S0_prem = init_price * np.exp(rho * integralstoch - 0.5 * integral * (rho ** 2))
    sig = np.sqrt((1 - (rho ** 2)) * integral / T)
    payoff = BlackScholes(S0_prem, K, r, sig, T)
    return payoff


def heston_conditional(mu, rho, kappa, sigmasquare0, theta, init_price,
                       sim_n, step_n, T, K):
    """ conditional Monte Carlo Milstein simulation
    sim_n: the number of simulations
    step_n: the number of steps
    """
    payoff = np.zeros([sim_n])
    std = (T / step_n) ** 0.5
    for i in range(0, sim_n):
        W = np.random.normal(0, std, step_n+1)
        payoff[i] = heston_condional_single(mu, rho, kappa, sigmasquare0,
                    theta, init_price, T, step_n, K, W)
    mean = np.mean(payoff)
    variance = np.var(payoff)
    return mean, variance


def heston_conditional_anti(mu, rho, kappa, sigmasquare0, theta, init_price,
                       sim_n, step_n, T, K):
    """ conditional Monte Carlo Milstein simulation with antithetic variate
    sim_n: the number of simulations
    step_n: the number of steps
    """
    payoff = np.zeros([sim_n])
    std = (T / step_n) ** 0.5
    for i in range(0, sim_n):
        W = np.random.normal(0, std, step_n+1)
        payoff1 = heston_conditional_single(mu, rho, kappa, sigmasquare0,
                    theta, init_price, T, step_n, K, W)
        W1 = -W
        payoff2 = heston_conditional_single(mu, rho, kappa, sigmasquare0,
                    theta, init_price, T, step_n, K, W1)
        payoff[i] = (payoff1 + payoff2)/2
    mean = np.mean(payoff)
    variance = np.var(payoff)
    return mean, variance


def heston_run(mu, rho, kappa, sigsquare0, theta,
               init_price, sim_n, step_n, T, K, r):
    num =50
    payoffs_heston = np.zeros([num])
    time1 = time.time()
    for i in tqdm(range(num)):
        mean_i, var_i = heston(mu, rho, kappa, sigsquare0,
                        theta, init_price, sim_n, step_n, T, K, r)
        payoffs_heston[i] = mean_i
    time2 = time.time()
    aveg_time = round((time2-time1) / num, 2)
    method_name = 'Normal Milstein'
    ls = [method_name, round(np.mean(payoffs_heston), 4), round(np.std(payoffs_heston), 4),
          aveg_time, sim_n, mu, rho, kappa, sigsquare0, theta]
    return ls

def heston_anti_run(mu, rho, kappa, sigsquare0, theta,
               init_price, sim_n, step_n, T, K, r):
    num =50
    payoffs_heston = np.zeros([num])
    time1 = time.time()
    for i in tqdm(range(num)):
        mean_i, var_i = heston_anti(mu, rho, kappa, sigsquare0,
                        theta, init_price, sim_n, step_n, T, K, r)
        payoffs_heston[i] = mean_i
    time2 = time.time()
    aveg_time = round((time2-time1) / num, 2)
    method_name = 'Normal Milstein + antithetic variate'
    ls = [method_name, round(np.mean(payoffs_heston), 4), round(np.std(payoffs_heston), 4),
          aveg_time, sim_n, mu, rho, kappa, sigsquare0, theta]
    return ls

def heston_control_run(mu, rho, kappa, sigsquare0, theta,
               init_price, sim_n, step_n, T, K, r):
    num =50
    corr = correlation(mu, rho, kappa, sigsquare0, theta,
                            init_price, sim_n, step_n, T, K, r)
    payoffs_heston = np.zeros([num])
    time1 = time.time()
    for i in tqdm(range(num)):
        mean_i, var_i = heston_control(mu, rho, kappa, sigsquare0,
                        theta, init_price, sim_n, step_n, T, K, r, corr)
        payoffs_heston[i] = mean_i
    time2 = time.time()
    aveg_time = round((time2-time1) / num, 2)
    method_name = 'Normal Milstein + control variate'
    ls = [method_name, round(np.mean(payoffs_heston), 4), round(np.std(payoffs_heston), 4),
          aveg_time, sim_n, mu, rho, kappa, sigsquare0, theta]
    return ls


def heston_condtional_run(mu, rho, kappa, sigsquare0, theta,
               init_price, sim_n, step_n, T, K, r):
    num =50
    payoffs_heston = np.zeros([num])
    time1 = time.time()
    for i in tqdm(range(num)):
        mean_i, var_i = heston_conditional(mu, rho, kappa, sigsquare0,
                        theta, init_price, sim_n, step_n, T, K)
        payoffs_heston[i] = mean_i
    time2 = time.time()
    aveg_time = round((time2-time1) / num, 2)
    method_name = 'Normal Milstein + conditional expectation'
    ls = [method_name, round(np.mean(payoffs_heston), 4), round(np.std(payoffs_heston), 4),
          aveg_time, sim_n, mu, rho, kappa, sigsquare0, theta]
    return ls



def heston_conditional_anti_run(mu, rho, kappa, sigsquare0, theta,
               init_price, sim_n, step_n, T, K, r):
    num =50
    payoffs_heston = np.zeros([num])
    time1 = time.time()
    for i in tqdm(range(num)):
        mean_i, var_i = heston_conditional_anti(mu, rho, kappa, sigsquare0,
                        theta, init_price, sim_n, step_n, T, K)
        payoffs_heston[i] = mean_i
    time2 = time.time()
    aveg_time = round((time2-time1) / num, 2)
    method_name = 'Normal Milstein + conditional expectation + antithetic variate'
    ls = [method_name, round(np.mean(payoffs_heston), 4), round(np.std(payoffs_heston), 4),
          aveg_time, sim_n, mu, rho, kappa, sigsquare0, theta]
    return ls

def run_experiment():
    # constant parameters:
    T = 1
    K = 100
    init_price = 100
    r = 0
    step_n = 1000
    sim_ns = [500, 1000, 2000]

    # change parameters:
    mus = [0.3, 0.4, 0.5]
    rhos = [-0.4, -0.5, -0.6]
    kappas = [0.5, 1, 1.5]
    sigsquare0s = [0.02, 0.03, 0.04]
    thetas = [0.05, 0.10, 0.15]
    resulsts_ls = []

    for i in range(3):
       for sim_n in sim_ns:
           mu_i = mus[i]
           rho_i = rhos[i]
           kappa_i = kappas[i]
           sigsquare0_i = sigsquare0s[i]
           theta_i = thetas[i]
           ls1 = heston_run(mu_i, rho_i, kappa_i, sigsquare0_i, theta_i,
                      init_price, sim_n, step_n, T, K, r)
           ls2 = heston_anti_run(mu_i, rho_i, kappa_i, sigsquare0_i, theta_i,
                      init_price, int(sim_n/2), step_n, T, K, r)
           ls3 = heston_control_run(mu_i, rho_i, kappa_i, sigsquare0_i, theta_i,
                           init_price, sim_n, step_n, T, K, r)
           ls4 = heston_condtional_run(mu_i, rho_i, kappa_i, sigsquare0_i, theta_i,
                           init_price, sim_n, step_n, T, K, r)
           ls5 = heston_conditional_anti_run(mu_i, rho_i, kappa_i, sigsquare0_i, theta_i,
                           init_price, int(sim_n/2), step_n, T, K, r)
           resulsts_ls.append(ls1)
           resulsts_ls.append(ls2)
           resulsts_ls.append(ls3)
           resulsts_ls.append(ls4)
           resulsts_ls.append(ls5)
    resulsts_ls = np.array(resulsts_ls)
    col_names = ['method name', 'call price mean', 'standard deviation',
                    'average time', 'simulation times', 'mu', 'rho', 'kappa', 'sigsquare0', 'theta']
    res_experiment = pd.DataFrame(data=resulsts_ls, columns=col_names)
    res_experiment.to_csv('exp_res.csv')
    return res_experiment




if __name__ == '__main__':
    run_experiment()