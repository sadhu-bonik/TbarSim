# Memresistor Crossbar Simulator

A small Python simulator for a 3D memristor crossbar array using nodal analysis. The main module is [`crossbar.py`](crossbar.py), which defines the `Tbar` class for building the graph, assigning conductances and bias voltages, solving the circuit, and visualizing the result in Plotly.

## Features

- Builds a 3D crossbar topology with wordlines, bitlines, vias, input nodes, and output nodes
- Supports bulk or per-device memristor conductance assignment
- Supports bulk or per-node input and output bias assignment
- Solves the network with sparse linear algebra
- Visualizes the topology and solved values in 3D

## Requirements

- Python 3.10+
- `numpy`
- `networkx`
- `plotly`
- `scipy`

## Install

```bash
python -m pip install numpy networkx plotly scipy
```

If you want to work in an isolated environment, create and activate a virtual environment first, then install the dependencies there.

## Quick Start

```python
from crossbar import Tbar

cb = Tbar(n=4, m=4, p=3)
cb.set_conductance(0, 0, 0, 55)
cb.randomize_conductances(seed=0)
cb.set_bias(vin=1.0, vout=0.0)
cb.solve()

print(cb.get_voltage("W(0,0,0)"))
print(cb.get_current("W(0,0,0)", "B(0,0,0)"))
```

## Run the Demo

The module includes a built-in demonstration block. Run it directly with:

```bash
python crossbar.py
```

## Notes

- `show()` opens an interactive Plotly figure.
- `solve()` must be called after conductances and bias voltages are fully specified.
- The current working tree uses the `Tbar` class name in `crossbar.py`.
