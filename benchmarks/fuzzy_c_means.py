from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from jax import random, jit, grad

@dataclass
class Galaxy:
    ra: float
    dec: float
    quantity_one: float
    quantity_two: float


class FCM:
    def __init__(self, n_clusters=2, m=2.0, max_iter=100, tol=1e-5):
        self.n_clusters = n_clusters
        self.m = m  # Fuzziness parameter
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.u = None

    def init_membership(self, key, num_of_points):
        key, subkey = random.split(key)
        self.u = random.uniform(subkey, shape=(num_of_points, self.n_clusters))
        self.u /= jnp.sum(self.u, axis=1, keepdims=True)

    def compute_cluster_centers(self, X):
        u_m = self.u ** self.m
        return (u_m.T @ X) / jnp.sum(u_m, axis=0)[:, jnp.newaxis]

    def update_membership(self, X):
        dist = jnp.linalg.norm(X[:, jnp.newaxis] - self.cluster_centers_, axis=2)
        dist = jnp.clip(dist, a_min=1e-10)  # Avoid division by zero
        dist = dist ** (2 / (self.m - 1))
        self.u = 1.0 / jnp.sum((dist[:, :, jnp.newaxis] / dist[:, jnp.newaxis, :]), axis=2)

    def fit(self, X, key=random.PRNGKey(0)):
        num_of_points = X.shape[0]
        self.init_membership(key, num_of_points)
        for i in range(self.max_iter):
            self.cluster_centers_ = self.compute_cluster_centers(X)
            prev_u = self.u.copy()
            self.update_membership(X)
            if jnp.linalg.norm(self.u - prev_u) < self.tol:
                break
            return self.u

    def predict(self, X):
        self.update_membership(X)
        return jnp.argmax(self.u, axis=1)

def compute_weights(X, key):
    fcm = FCM(n_clusters=2, m=2.0, max_iter=100)
    weight_matrix = fcm.fit(X, key)
    return weight_matrix

galaxies = [
    Galaxy(10.0, 20.0, 0.1, 0.2),
    Galaxy(12.0, 22.0, 0.3, 0.4),
    Galaxy(15.0, 24.0, 0.5, 0.6),
    Galaxy(30.0, 40.0, 0.7, 0.8),
    Galaxy(32.0, 42.0, 0.9, 1.0)
]

# Extract positions (ra and dec) as input data
positions = jnp.array([[galaxy.ra, galaxy.dec] for galaxy in galaxies], dtype=jnp.float32)

# Compute the weight matrix
key = random.PRNGKey(0)
weight_matrix = compute_weights(positions, key)

# Compute the gradient of the weight matrix with respect to the positions
grad_weights = grad(lambda X: jnp.sum(compute_weights(X, key)))(positions)

print("Weight Matrix (Membership Matrix):\n", weight_matrix)
print("Weight Matrix Gradient:\n", grad_weights)
