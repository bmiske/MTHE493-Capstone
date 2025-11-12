# MTHE493-Capstone

## Local setup

Create an isolated virtual environment so that `numpy` and `matplotlib` can be installed with compatible binary wheels instead of mixing the system packages (which triggered the `numpy.core.multiarray failed to import` error when `/bin/python` was used).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the simulation

After the environment is activated, invoke the driver with the interpreter from that environment:

```bash
python TrialSimulation.py
```

Avoid calling `/bin/python` because it bypasses the virtual environment and re-introduces the incompatible system packages that are compiled against NumPy 1.x.
