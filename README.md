# 🏓 **PONG DQN** 🎮
<p align="center">
  <img src="https://img.shields.io/badge/Python-blue" alt="Python 3.10">
  <img src="https://img.shields.io/badge/Reinforcement_Learning-DQN-green" alt="DQN">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="PyTorch">
</p>

## 📽️ **Demonstration**
<p align="center">
  <video autoplay loop muted playsinline width="300">
    <source src="src/videos/pong_dqn.gif" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>

---

## 🌟 **Introduction**
Welcome to **PONG DQN**, a project using Deep Q-Network (DQN) reinforcement learning to train an agent to master the game Pong. The project combines modern techniques such as Dual DQN, replay buffer, and epsilon-greedy strategy for advanced optimization. 🚀

---

## 🛠️ **Key Features**
- **Image preprocessing**: reduction and conversion to grayscale for increased efficiency.
- **DQN Algorithm**: learning based on optimal states and actions.
- **Dual DQN Architecture**: improved stability and performance through the separation of value and advantage streams.
- **Two Networks Management**: Policy and Target for robust convergence.

---

## 🚀 **Installation**
Follow these steps to set up and run the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/eithannak29/Pong-DQN.git
   cd Pong-DQN
   ```

2. **Install dependencies**:
   Using **`uv`**:
   ```bash
   uv sync
   ```

3. **Run training or demonstration**:
   Open the notebook:
   ```bash
   jupyter notebook src/main.ipynb
   ```

---

## 📊 **Results**
The agent was trained in 1h43 on a NVIDIA RTX 4070. Here is a summary of the results:
- **Cumulative Rewards**: Stabilized at 20, indicating an effective strategy.  
- **Number of Steps**: Plateau after 300 episodes.  
- **Epsilon and Total Loss**: Significant reduction, demonstrating successful learning.  

---

## 💡 **Methodology**
1. Image preprocessing (grayscale, resize 80x64, normalization).  
2. Training with Dual DQN and replay buffer.  
3. Management of Policy and Target networks for increased stability.  

---

## 🤝 **Contributors**
- **Camil Ziane**
- **Eithan Nakache**  

