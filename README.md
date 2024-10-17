
## Validation experiments for SoftCVI

Experiments for validating performance of the Soft contrastive variational inference, as implemented in [pyrox](https://github.com/danielward27/pyrox).

- For a description of the method, see the [arxiv paper](https://arxiv.org/pdf/2407.15687).
- We use HPC for running the experiments, see [jobs](jobs/).
- Single runs can be performed using [scripts/run_task.py](scripts/run_task.py), for example by running:
```python
python -m scripts.run_task --seed=0 --task-name="eight_schools"
```

## Constructing the environment
The exact conda environment used in the experiments is in [softcvi.yml](softcvi.yml).
However, this likely will not be compatible across platforms. A less precise, but more
convenient way to recreate the environment is to clone
[pyrox](https://github.com/danielward27/pyrox)  and
``softcvi_validation`` into your current directory, and run:
```
conda create --name pyrox_env python
conda activate pyrox_env
pip install -e softcvi
pip install -e softcvi_validation
```