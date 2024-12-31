
## Validation experiments for SoftCVI

Experiments for validating performance of Soft Contrastive Variational Inference (SoftCVI):

- For a description of the method, see the [arxiv paper](https://arxiv.org/pdf/2407.15687).
- The implementation is in [pyrox](https://github.com/danielward27/pyrox).
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
conda create --name softcvi_validation_env python
conda activate softcvi_validation_env
pip install -e pyrox
pip install -e softcvi_validation
```
