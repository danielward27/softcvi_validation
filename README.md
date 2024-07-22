
## Validation experiments for SoftCVI

Experiments for validating performance of [softcvi](https://github.com/danielward27/softcvi).
This package requires softcvi to be installed.

- We use HPC for running the experiments, see [jobs](jobs/).
- Single runs can be performed using [scripts/run_task.py](scripts/run_task.py), for example by running:
```python
python -m scripts.run_task --seed=0 --task-name="eight_schools"
```
