#%%
import matplotlib.pyplot as plt
from BayesForge import bf
import jax.numpy as jnp
import numpy as np
import jax
import os, sys

# The script is run with Test/10.BNN as working directory (run_all.py), but its
# data/ and results/ folders are located from the script itself so that it also
# runs from its own directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))

sys.path.insert(0, os.path.join(_HERE, "..", "..", "3.Documentation"))
from doc_blocks import doc_model

# The model as published on the documentation page.
# Same statistical model and the same priors as the copy this script carried
# (w1 ~ normal(0, 1) of shape (D_X, D_H) with tanh, w2 ~ normal(0, 1) of shape
# (D_H, D_Y), sigma ~ exponential(1), Y ~ normal(w2, sigma)); the only
# difference is the name of the likelihood scale: the page calls it 'sigma',
# the old copy 's'.
# The page opens 'BNN.json' from the working directory, so the page is loaded
# with the folder holding that file (Test/10.BNN/data) as working directory.
_CWD = os.getcwd()
os.chdir(os.path.join(_ROOT, "data"))
PAGE_MODEL = doc_model("33. BNN.qmd", 0)
os.chdir(_CWD)

m = bf(platform='cpu')

# %%
# setup platform------------------------------------------------
m = bf(platform='cpu')
alpha= 0.6
beta = 0.5
sigma= 5
x=m.dist.normal(160, 40,sample=True,shape=(100,))
lk=alpha+beta*x
y=m.dist.normal(lk, sigma,sample=True,seed=2)
data=jnp.array([x,y]).T

def split_train_test(arr):
    # Get the length of the array
    n = arr.shape[0]
    
    # Calculate the split index (80% for training)
    train_idx = int(n * 0.8)
    
    # Generate a random permutation of indices
    key = jax.random.PRNGKey(0)  # You can use a fixed key for reproducibility
    permuted_indices = jax.random.permutation(key, n)
    
    # Split the permuted indices
    train_indices = permuted_indices[:train_idx]
    test_indices = permuted_indices[train_idx:]
    
    # Split the original array based on the indices
    train_set = arr[train_indices]
    test_set = arr[test_indices]
    
    return train_set, test_set

train, test = split_train_test(data)

dataTrain= dict(X=train[:,0].reshape((train[:,0].shape[0],1)),Y=train[:,1].reshape((train[:,0].shape[0],1)))

m.data_on_model=dataTrain

plt.scatter(dataTrain['X'], dataTrain['Y'])
# %%
X_mean = jnp.mean(dataTrain['X'], axis=0)
X_std = jnp.std(dataTrain['X'], axis=0)
Y_mean = jnp.mean(dataTrain['Y'], axis=0)
Y_std = jnp.std(dataTrain['Y'], axis=0)
X=(dataTrain['X'] - X_mean) / X_std
Y=(dataTrain['Y'] - Y_mean) / Y_std
m.data_on_model= dict(X=X, Y=Y)


# %%
# The BNN is the one published on the page (PAGE_MODEL): one tanh hidden layer of
# D_H = 5 units, linear output layer, exponential(1) scale on the Normal likelihood.
m.fit(PAGE_MODEL, num_samples=500) 

# --- CORRECTED PLOTTING CODE ---
pred_standardized = m.sample(samples = 500)['Y']
# Calculate statistics on the standardized predictions
mean_pred_std = jnp.mean(pred_standardized, axis=0).squeeze()
percentiles_std = np.percentile(pred_standardized, [5.0, 95.0], axis=0).squeeze()


# --- 2. De-standardize Predictions ---
# This is the critical step. We use the mean and std deviation from the 
# ORIGINAL training data (Y_mean, Y_std) to scale the predictions back.
mean_pred_orig = (mean_pred_std * Y_std) + Y_mean
percentiles_orig = (percentiles_std * Y_std) + Y_mean


# --- 3. Prepare Data for Plotting ---
# For a clean line plot, it's essential to sort the X values.
# We must apply the same sorting order to our predictions and percentiles.
X_orig = dataTrain['X'].squeeze()
Y_orig = dataTrain['Y'].squeeze()

sort_indices = jnp.argsort(X_orig)

X_plot = X_orig[sort_indices]
Y_plot = Y_orig[sort_indices]
mean_pred_plot = mean_pred_orig[sort_indices]
percentiles_plot = percentiles_orig[:, sort_indices]


# --- 4. Plotting on Original Scale ---
# Now, we use the original data and the de-standardized predictions.
fig, ax = plt.subplots(figsize=(10, 7))

# Plot original training data
ax.plot(X_plot, Y_plot, 'kx', label='Training Data')

# Plot the 90% Credible Interval (de-standardized)
ax.fill_between(
    X_plot,
    percentiles_plot[0, :],
    percentiles_plot[1, :],
    color='lightblue',
    label='90% Credible Interval'
)

# Plot the mean prediction (de-standardized)
ax.plot(X_plot, mean_pred_plot, 'b-', lw=2, label='Mean Prediction')

# Set labels, title, and legend
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("BNN Mean predictions with 90% CI (Corrected)")
ax.legend()
ax.set_ylim(top=140) # Set y-limit to match example image

plt.show()
# save plot
plt.savefig(os.path.join(_ROOT, "results", "bnn_prediction_linear.png"))
# %%
def get_data(N=50, D_X=1, sigma_obs=0.05, N_test=500):
    D_Y = 1  # create 1d outputs
    np.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    X = jnp.power(X[:, np.newaxis], jnp.arange(D_X))
    W = 0.5 * np.random.randn(D_X)
    Y = jnp.dot(X, W) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])
    Y += sigma_obs * np.random.randn(N)
    Y = Y[:, np.newaxis]
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)

    X_test = jnp.linspace(-1.3, 1.3, N_test)
    X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))

    return X, Y, X_test
X, Y, X_test = get_data(D_X = 2)



X=(X - jnp.mean(X))/jnp.std(X)
m.data_on_model=dict(X=X, Y=Y) 

import json
with open(os.path.join(_ROOT, "data", "BNN.json"), "w") as outfile:
    json.dump(dict(X=X.tolist(), Y=Y.tolist()), outfile)
m.fit(PAGE_MODEL, num_samples=500)   
#%%
pred = m.sample(samples = 500)['Y']
pred = pred[..., 0]
mean_prediction = jnp.mean(pred, axis=0)
percentiles = np.percentile(pred, [5.0, 95.0], axis=0)
# make plots
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
# plot training data
ax.plot(X[:, 1], Y[:, 0], "kx")
# plot 90% confidence level of predictions
ax.fill_between(
    X[:, 1], percentiles[0, :], percentiles[1, :], color="lightblue"
)
# plot mean prediction
ax.plot(X[:, 1], mean_prediction, "blue", ls="solid", lw=2.0)
ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")

#save plot
fig.savefig(os.path.join(_ROOT, "results", "bnn_prediction_non_linear.png"))
