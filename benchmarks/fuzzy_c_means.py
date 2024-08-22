from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from jax import random, jit, grad, value_and_grad

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

    def create_new_galaxies(self, X, quantities):
        new_galaxies = []
        for c in range(self.n_clusters):
            centroid = self.cluster_centers_[c]
            u_m = self.u[:, c] ** self.m
            weighted_quantities = (u_m[:, None] * quantities).sum(axis=0) / u_m.sum()

            new_galaxy = jnp.array([
                centroid[0],  # ra
                centroid[1],  # dec
                weighted_quantities[0],  # quantity_one
                weighted_quantities[1]   # quantity_two
            ])
            new_galaxies.append(new_galaxy)

        return jnp.array(new_galaxies)

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

def compute_new_galaxies(positions, quantities, key):
    fcm = FCM(n_clusters=2, m=2.0, max_iter=100)
    fcm.fit(positions, key)
    new_galaxies = fcm.create_new_galaxies(positions, quantities)
    return new_galaxies

def compute_component(positions, quantities, key, idx, component_idx):
    fcm = FCM(n_clusters=2, m=2.0, max_iter=100)
    fcm.fit(positions, key)
    new_galaxies = fcm.create_new_galaxies(positions, quantities)
    return new_galaxies[idx, component_idx]

# Use JAX's grad to get the derivatives
key = random.PRNGKey(0)
positions = jnp.array([[galaxy.ra, galaxy.dec] for galaxy in galaxies], dtype=jnp.float32)
quantities = jnp.array([[galaxy.quantity_one, galaxy.quantity_two] for galaxy in galaxies], dtype=jnp.float32)

# Select a specific galaxy to compute gradients with respect to
galaxy_idx = 0  # Index of the first galaxy
for component_idx, component_name in enumerate(["ra", "dec", "quantity_one", "quantity_two"]):
    grad_func = grad(lambda p, q: compute_component(p, q, key, galaxy_idx, component_idx))
    galaxy_grads = grad_func(positions, quantities)
    print(f"Gradients of {component_name} with respect to positions:\n", galaxy_grads[0])
    print(f"Gradients of {component_name} with respect to quantities:\n", galaxy_grads[1])
