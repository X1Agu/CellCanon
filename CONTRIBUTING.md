# Contributing

Follow [the environment guide](environments/README.md), then install the
development dependencies and run the tests:

```sh
python -m pip install '.[dev]'
python -m pytest -q
```

Describe the problem, changes and relevant tests in each pull request.
Changes to scientific protocols should explain the effect on the experiment.
Keep credentials, datasets, weights and generated outputs out of commits.
Use synthetic inputs in tests and redact credentials from issue reports.
