# GRAN-DAG Play

This repository contains scripts to generate data and train a model using a non-linear Gaussian additive noise model.

## Getting Started

### Prerequisites

Ensure you have the necessary dependencies installed. You can install them using:

```sh
pip install -r requirements.txt
```

### Generating Data

To generate data, use the `generate_data.sh` script. You can control the number of DAGs and the mode (training or test data) with the following options:

- `--num_graphs`: Number of DAGs to generate
- `--mode`: Mode for data generation (`train` or `test`)

Example usage:

```sh
./generate_data.sh --num_graphs 100 --mode train
```

### Training the Model

To train the model, use the `train_gran_dag.sh` script. All hyperparameters can be found and adjusted inside the script.

Example usage:

```sh
./train_gran_dag.sh
```
