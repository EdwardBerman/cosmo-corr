import jax.numpy as jnp
from dataclasses import dataclass
from typing import List
from astropy.io import fits

@dataclass
galaxy:
    ra: float
    dec: float
    quantity_one: float
    quantity_two: float


def differentiable_kmeans(data: List[galaxy], galaxy, n_clusters: int, max_iters=100 :int, tol=1e-4 :float): -> jnp.ndarray:
    min_ra = min([galaxy.ra for galaxy in data])
    max_ra = max([galaxy.ra for galaxy in data])
    min_dec = min([galaxy.dec for galaxy in data])
    max_dec = max([galaxy.dec for galaxy in data])

    cluster_centers = jnp.array([[jnp.random.uniform(min_ra, max_ra), jnp.random.uniform(min_dec, max_dec)] for _ in range(n_clusters)])
    for _ in range(max_iters):
        for i in range(len(data)):
            distances = jnp.linalg.norm(cluster_centers - jnp.array([data[i].ra, data[i].dec]), axis=1)
            cluster = jnp.argmin(distances)
            cluster_centers[cluster] = jnp.mean(jnp.array([data[i].ra, data[i].dec]), axis=0)

    return cluster_centers


