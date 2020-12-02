import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import matplotlib as mpl


def plot_ellipses(ax, weights, means, covars):
    for n in range(means.shape[0]):
        eig_vals, eig_vecs = np.linalg.eigh(covars[n])
        unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
        angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
        # Ellipse needs degrees
        angle = 180 * angle / np.pi
        # eigenvector normalization
        eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
        ell = mpl.patches.Ellipse(means[n], eig_vals[0], eig_vals[1],
            180 + angle, edgecolor='black')
        #ell.set_clip_box(ax.bbox)
        ell.set_alpha(weights[n])
        #ell.set_facecolor('#56B4E9')
        ell.set_edgecolor('k')
        ax.add_artist(ell)


# set random seed
np.random.seed(0)

# number of clusters
n_clusters = 4

# designate cluster centers
cluster_centers = np.array([
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10]
])

# sample cluster probabilities
cluster_probs = np.random.dirichlet(
    np.ones(n_clusters), 1)

# number of samples to draw from clusters
n_samples = 1000

# sample clusters
cluster_samples = np.random.choice(
    range(n_clusters), n_samples, p=cluster_probs.flatten())

# noise
noise = np.random.normal(size=(n_samples, cluster_centers.shape[1]))

# draw samples
samples = np.zeros((n_samples, cluster_centers.shape[1]))
for i in range(n_samples):
    samples[i] = cluster_centers[cluster_samples[i]] + noise[i]

# cluster
dp = mixture.GaussianMixture(n_components=10)
labels = dp.fit_predict(samples)

# plot
plt.scatter(samples[:, 0], samples[:, 1], c=labels)
plot_ellipses(plt.gca(), dp.weights_, dp.means_, dp.covariances_)
plt.show()
