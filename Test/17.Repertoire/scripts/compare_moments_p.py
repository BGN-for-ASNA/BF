import pandas as pd
import numpy as np
import json
import os
import sys
from scipy.stats import skew, kurtosis

# Add BF to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from BayesForge import bf

def main():
    with open('../data/stan_data.json', 'r') as f:
        data = json.load(f)
    Y_int = np.array(data['Y'], dtype=np.int32)
    BF_data = {'N': data['N'], 'M': data['M'], 'J': data['J'], 'd': np.array(data['d']), 'Y': Y_int}
    
    stan_samples = pd.read_csv('../data/stan_reference_samples.csv')
    
    m = bf('cpu')
    def BF_model(**d):
        L = m.dist.exponential(1.0, shape=(d['M'],), name='L')
        p = m.dist.beta(2.0, 2.0, shape=(d['M'],), name='p')
        rate = L[None, :] * d['d'][:, None]
        gate = 1.0 - p[None, :]
        m.dist.zero_inflated_poisson(gate=gate, rate=rate, obs=d['Y'], name='Y')

    m.fit(BF_model, obs=BF_data, num_samples=1000, num_warmup=1000, num_chains=2, seed=42)
    BF_post = m.posteriors
    
    print(f"{'Metric':<15} | {'Stan (p[1])':<15} | {'BF (p[0])':<15} | {'Diff':<15}")
    print("-" * 65)
    
    # BF index i (0-based) maps to Stan column p[i+1] (1-based).
    for idx_BF in [0, 1]:
        idx_stan = idx_BF + 1
        s_vals = stan_samples[f'p[{idx_stan}]'].values
        b_vals = BF_post['p'][:, idx_BF]
        
        metrics = [
            ('Mean', np.mean),
            ('SD', np.std),
            ('Skew', skew),
            ('Kurtosis', kurtosis),
            ('Min', np.min),
            ('Max', np.max)
        ]
        
        print(f"\nComparing BF:p[{idx_BF}] vs Stan:p[{idx_stan}]")
        for name, func in metrics:
            s_m = func(s_vals)
            b_m = func(b_vals)
            print(f"{name:<15} | {s_m:15.4f} | {b_m:15.4f} | {s_m - b_m:15.4f}")

        # z-score of the mean difference in sampling-error units
        z = abs(np.mean(b_vals) - np.mean(s_vals)) / np.sqrt(
            np.std(b_vals) ** 2 / len(b_vals) + np.std(s_vals) ** 2 / len(s_vals))
        print(f"z-score of mean difference: {z:.2f}")
        assert z < 4, f"BF:p[{idx_BF}] vs Stan:p[{idx_stan}]: z = {z:.2f} >= 4"

if __name__ == "__main__":
    main()
