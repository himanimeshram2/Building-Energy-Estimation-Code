#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install numpy random gym matplotlib scikit-learn deap multiprocessing seaborn tensorflow tqdm dask pandas pyarrow')


# In[ ]:


import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from deap import base, creator, tools, algorithms
import multiprocessing
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from deap.tools import ParetoFront
from tqdm import tqdm
import dask.dataframe as dd
import pandas as pd
import pyarrow.feather as feather
import os

# Load all sensor data CSV files from a local folder using Dask
import glob

sensor_data_folder = r"C:\Users\Hp\Downloads\fwlmb11wni392kodtyljkw4n2\fwlmb11wni392kodtyljkw4n2\files_csv"
sensor_data_files = glob.glob(os.path.join(sensor_data_folder, '*.csv'))

# Load sensor data using Dask
sensor_data = dd.read_csv(sensor_data_files)

# Check available columns and rename if necessary
print("Available columns in sensor data:", sensor_data.columns)

if 'time' not in sensor_data.columns:
    # Rename the appropriate column to 'time' if it exists with a different name
    sensor_data = sensor_data.rename(columns={"actual_time_column_name": "time"})

# Optionally compute to reduce the dataset size, for example, calculate means or downsample
# This can reduce memory usage before converting to Pandas DataFrame for further processing
if 'time' in sensor_data.columns:
    sensor_data_reduced = sensor_data.groupby(sensor_data['time']).mean().compute()
else:
    print("The 'time' column is not found in the sensor data files. Proceeding without grouping.")
    sensor_data_reduced = sensor_data.compute()

# Load Building Metadata from a local file or accessible link
building_metadata_csv_path = r"C:\Users\Hp\Downloads\building_metadata.csv"
building_metadata_feather_path = r"C:\Users\Hp\Downloads\building_metadata.feather"

building_metadata_csv_df = pd.read_csv(building_metadata_csv_path)
try:
    building_metadata_feather_df = feather.read_feather(building_metadata_feather_path)
except Exception as e:
    print(f"Error loading feather file: {e}")
    building_metadata_feather_df = None

# Load Structural Model, Measurement Model, and Supplementary Figure Dataset from local files or accessible links
structural_model_csv_path = r"C:\Users\Hp\Downloads\Structural Model.xlsx"
measurement_model_csv_path = r"C:\Users\Hp\Downloads\Measurement Model.xlsx"
supplementary_figure_dataset_path = r"C:\Users\Hp\Downloads\Supplementary_Figure_Dataset.xlsx"

structural_model_df = pd.read_excel(structural_model_csv_path)
measurement_model_df = pd.read_excel(measurement_model_csv_path)
supplementary_figure_dataset_df = pd.read_excel(supplementary_figure_dataset_path)

# Load Sample CSV
sample_csv_path = r"C:\Users\Hp\Downloads\Sample.csv"
sample_df = pd.read_csv(sample_csv_path)

# Example: Extracting occupancy data and external temperature from metadata
occupancy_data = building_metadata_csv_df['occupancy'] if 'occupancy' in building_metadata_csv_df else None
external_temp_data = building_metadata_csv_df['external_temperature'] if 'external_temperature' in building_metadata_csv_df else None

# Environment Setup
class BuildingEnergyEnv(gym.Env):
    def __init__(self):
        super(BuildingEnergyEnv, self).__init__()
        
        # Define action and observation space
        # Actions: [-1, 1] -> [-1 = decrease HVAC, 1 = increase HVAC]
        self.action_space = spaces.Discrete(3)  # Actions: 0 (Decrease), 1 (Maintain), 2 (Increase)
        # Observation space: [Temperature, Occupancy, External Temperature]
        self.observation_space = spaces.Box(low=np.array([0, 0, -30]), high=np.array([50, 1, 50]), dtype=np.float32)
        
        # Set initial state
        self.state = [22.0, 1, 25.0]  # Indoor temperature, Occupancy, Outdoor temperature
        self.episode_length = 24
        self.current_step = 0
        
        # Load data for environment dynamics
        self.occupancy_data = occupancy_data.values if occupancy_data is not None else [1] * self.episode_length
        self.external_temp_data = external_temp_data.values if external_temp_data is not None else [25.0] * self.episode_length
        
        # Energy use parameters
        self.energy_consumption = 0
        self.total_reward = 0
    
    def reset(self):
        # Reset state
        self.state = [22.0, 1, 25.0]
        self.current_step = 0
        self.energy_consumption = 0
        self.total_reward = 0
        return np.array(self.state, dtype=np.float32)
    
    def step(self, action):
        indoor_temp, occupancy, outdoor_temp = self.state
        self.current_step += 1
        
        # Update state from real data if available
        if self.current_step < len(self.occupancy_data):
            occupancy = self.occupancy_data[self.current_step]
            outdoor_temp = self.external_temp_data[self.current_step]
        
        # Update indoor temperature based on action
        if action == 0:  # Decrease HVAC
            indoor_temp -= 1.0
        elif action == 2:  # Increase HVAC
            indoor_temp += 1.0
        
        # Apply external temperature influence
        indoor_temp += 0.1 * (outdoor_temp - indoor_temp)
        
        # Calculate reward (comfort + energy saving)
        comfort_penalty = -abs(indoor_temp - 22.0) if occupancy == 1 else 0
        energy_penalty = -abs(action) * 0.1
        reward = comfort_penalty + energy_penalty
        
        self.total_reward += reward
        self.energy_consumption += abs(action) * 0.1
        
        # End condition
        done = self.current_step >= self.episode_length
        
        # Update state
        self.state = [indoor_temp, occupancy, outdoor_temp]
        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    def render(self):
        print(f"Step: {self.current_step}, State: {self.state}, Total Reward: {self.total_reward}, Energy Consumption: {self.energy_consumption}")

# Artificial Neural Network Model for Predicting Energy Consumption
class EnergyANN:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=1)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

# Genetic Algorithm Optimization using DEAP with Multi-objective NSGA-II
def evaluate(individual):
    hvac_settings = individual
    env = BuildingEnergyEnv()
    total_reward = 0
    total_energy_consumption = 0
    total_comfort_penalty = 0
    state = env.reset()
    for setting in hvac_settings:
        action = setting
        state, reward, done, _ = env.step(action)
        total_reward += reward
        total_energy_consumption += abs(action) * 0.1
        total_comfort_penalty += -abs(state[0] - 22.0) if state[1] == 1 else 0
        if done:
            break
    return total_reward, total_energy_consumption, total_comfort_penalty

creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=24)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

# Deep Q-Learning Agent with Neural Network
class DeepQLearningAgent:
    def __init__(self, env):
        self.env = env
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration-exploitation trade-off
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.memory = []
        self.batch_size = 32
        self.update_target_frequency = 10

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.env.observation_space.shape[0], activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.model.predict(np.array([state]))
            return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train(self, episodes):
        rewards = []
        for episode in tqdm(range(episodes), desc="Training Episodes"):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.replay()

            rewards.append(total_reward)
            if episode % self.update_target_frequency == 0:
                self.update_target_model()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")
        return rewards

# Run the environment, agent, and GA
if __name__ == "__main__":
    # Train Deep Q-Learning Agent
    env = BuildingEnergyEnv()
    agent = DeepQLearningAgent(env)
    episodes = 5  # Reduced the number of episodes for quicker demonstration
    rewards = agent.train(episodes)

    # Plot rewards
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=rewards, label='Total Reward', color='b')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Run NSGA-II Optimization
    population_size = 10  # Reduced population size for quicker demonstration
    generations = 5  # Reduced generations for quicker demonstration
    population = toolbox.population(n=population_size)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    pareto_front = ParetoFront()
    algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=population_size, cxpb=0.7, mutpb=0.3, ngen=generations, 
                              stats=None, halloffame=pareto_front, verbose=True)

    # Plot Pareto front
    pareto_rewards = [ind.fitness.values[0] for ind in pareto_front]
    pareto_energy = [ind.fitness.values[1] for ind in pareto_front]

    plt.figure(figsize=(10, 6))
    plt.scatter(pareto_energy, pareto_rewards, alpha=0.7, c='r')
    plt.xlabel('Energy Consumption (kWh)')
    plt.ylabel('Total Reward')
    plt.title('Pareto Front for Energy Consumption vs Reward')
    plt.grid(True)
    plt.show()


# In[ ]:





