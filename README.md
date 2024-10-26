# Building Energy Estimation

This project provides a comprehensive solution for estimating energy consumption in buildings using sensor data. The model employs machine learning techniques to analyze historical data and predict future energy usage patterns.

## Features

- **Data Loading**: Utilizes Dask for efficient handling of large CSV files containing sensor data.
- **Machine Learning**: Implements a Multi-Layer Perceptron (MLP) regressor for energy estimation.
- **Optimization**: Includes genetic algorithms for parameter tuning and performance optimization.
- **Visualization**: Provides various plots using Matplotlib and Seaborn to visualize energy consumption patterns.
- **Parallel Processing**: Uses multiprocessing to enhance computational efficiency.

## Requirements

Before running the code, ensure you have the following Python packages installed:

pip install numpy random gym matplotlib scikit-learn deap multiprocessing seaborn tensorflow tqdm dask pandas pyarrow

### Getting Started

1. Clone the repository:
   
git clone https://github.com/your_username/Building_Energy_Estimation.git
cd Building_Energy_Estimation

2. Set up your sensor data: Place all your sensor data CSV files in the files_csv folder. Update the sensor_data_folder variable in the code to point to your local directory.

- Run the script: Execute the main script to start the energy estimation process:

python Building_Energy_Estimation.py

### Code Structure

Building_Energy_Estimation.py: Main script for loading data, training the model, and making predictions.

requirements.txt: List of dependencies for easy installation.

### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgements

- TensorFlow
- Dask
- scikit-learn
- DEAP


### Instructions:

- Replace `your_username` in the clone URL with your actual GitHub username.
- Adjust any sections to better fit the specific functionalities and features of your code as you see fit. If there are additional features or details about your project, feel free to include those in the README as well. &#8203;:contentReference[oaicite:0]{index=0}&#8203;
