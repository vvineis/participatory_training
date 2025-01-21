# Participatory Training Framework

To set up your environment, follow these steps:

1. **Create a Conda Environment**
   Run the following command to create a new Conda environment with Python 3.11.9:
   ```bash
   conda create --name PartTrainEnv python=3.11.9

2. **Install packages**
   ```bash
   pip install -r requirements.txt

3. **Running the main code**
    ```bash
    python main.py

### Customizing Configurations
To specify custom configurations, modify the parameters directly in the command. For example:
    ```bash
    python main.py use_case=lending cv_splits=5 sample_size=10000

Refer to the configuration files for detailed information about the available parameters. 

## Adaptation to New Use Cases

This code can be adapted to new use cases and datasets with slight modifications. 
Refer to the comments and documentation in the codebase to identify where changes are required for incorporating your specific application.

## Experiments presented in the paper
The outputs of the experiments presented in the paper are located in the "results" folder. This folder also contains the code used to generate the plots.
