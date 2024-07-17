# %%
import bz2
import io
import json
import zipfile

import numpy as np
import requests

# By convention the posteriors have shape (n_obs, n_samps, *latent_shape)
# and the observations have shape (n_obs, *obs_shape).
# The keys match the names of the nodes in the model.


def get_slcp_posterior():
    obs = []
    latents = []

    for i in range(1, 11):
        url = f"https://github.com/sbi-benchmark/sbibm/raw/v1.1.0/sbibm/tasks/slcp/files/num_observation_{i}"

        # Load in observations
        response = requests.get(f"{url}/observation.csv")
        response.raise_for_status()
        csv_content = io.StringIO(response.text)
        obs.append(
            np.genfromtxt(csv_content, delimiter=",", skip_header=1).reshape(4, 2),
        )

        # Load in reference samples
        response = requests.get(f"{url}/reference_posterior_samples.csv.bz2")
        response.raise_for_status()
        decompressed_content = bz2.decompress(response.content)
        csv_content = io.StringIO(decompressed_content.decode("utf-8"))
        latents.append(np.genfromtxt(csv_content, delimiter=",", skip_header=1))

    latents, obs = np.stack(latents), np.stack(obs)
    np.savez("reference_posteriors/slcp/latents.npz", theta=latents)
    np.savez("reference_posteriors/slcp/observations.npz", x=obs)


def get_posteriordb_posterior(name):
    url = f"https://github.com/stan-dev/posteriordb/raw/0.5.0/posterior_database/reference_posteriors/draws/draws/{name}.json.zip"

    response = requests.get(url)
    response.raise_for_status()
    zip_content = io.BytesIO(response.content)

    with zipfile.ZipFile(zip_content, "r") as zip_ref:
        with zip_ref.open(zip_ref.infolist()[0]) as json_file:
            draws = json.load(json_file)

    # Concatenate the chains
    draws = {
        k: np.concatenate([np.asarray(chain[k]) for chain in draws])
        for k in draws[0].keys()
    }

    # Names may be of form param[1] param[2]. We want to stack these into an array
    stacked_draws = {k.split("[")[0]: [] for k in draws.keys()}
    for k, v in draws.items():
        key_root = k.split("[")[0]
        stacked_draws[key_root].append(v)
    return {
        k: np.stack(v, axis=-1).squeeze()[np.newaxis, ...]
        for k, v in stacked_draws.items()
    }


def get_posteriordb_data(name):
    url = f"https://github.com/stan-dev/posteriordb/raw/0.5.0/posterior_database/data/data/{name}.json.zip"
    response = requests.get(url)
    response.raise_for_status()
    zip_content = io.BytesIO(response.content)

    with zipfile.ZipFile(zip_content, "r") as zip_ref:
        with zip_ref.open(zip_ref.infolist()[0]) as json_file:
            data = json.load(json_file)
    return {k: np.array(arr, dtype=float) for k, arr in data.items()}


def get_eight_schools_posterior():
    latents = get_posteriordb_posterior("eight_schools-eight_schools_noncentered")
    obs = {"y": get_posteriordb_data("eight_schools")["y"][np.newaxis, ...]}
    np.savez("reference_posteriors/eight_schools/latents.npz", **latents)
    np.savez("reference_posteriors/eight_schools/observations.npz", **obs)


def get_garch_posterior():
    latents = get_posteriordb_posterior("garch-garch11")
    obs = {"y": get_posteriordb_data("garch")["y"][np.newaxis, ...]}
    np.savez("reference_posteriors/garch/latents.npz", **latents)
    np.savez("reference_posteriors/garch/observations.npz", **obs)


if __name__ == "__main__":
    # Run with python -m scripts.get_reference_posteriors
    get_eight_schools_posterior()
    get_slcp_posterior()
    get_garch_posterior()

# %%
