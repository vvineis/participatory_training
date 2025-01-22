# Participatory Training Framework

To set up your environment and run the experiments follow this steps:

1. **Create a Conda Environment** 
   ```bash
   conda create --name PartTrainEnv python=3.11.9

2. **Install packages**  
   ```bash
   pip install -r requirements.txt
   
4. **Modify the path**  
Update the paths in the .yaml configuration files located in conf>use_case to match your setup. Specifically, modify the data_path and result_path parameters in the lending and health configuration files to point to the correct directories on your system.

5. **Running the main code**
    ```bash
    python main.py

6. **Customizing experiments configuration**  
To specify custom configurations, modify the parameters directly in the command. For example,
   ```bash
   python main.py use_case=lending cv_splits=5 sample_size=10000

Refer to the configuration files for detailed information about the available parameters. 

## Adaptation to New Use Cases

This code can be adapted to new use cases and datasets with slight modifications. 
Refer to the comments and documentation in the codebase to identify where changes are required for incorporating your specific application.

## Experiments presented in the paper
The outputs of the experiments presented in the paper are located in the "results" folder. This folder also contains the code used to generate the plots.
