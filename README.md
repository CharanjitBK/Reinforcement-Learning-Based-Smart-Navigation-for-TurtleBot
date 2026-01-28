# Reinforcement Learning Based Smart Navigation for Turtlebot (ROS + Gazebo)

This repository implements **autonomous mobile robot navigation using Reinforcement Learning (RL)** in **ROS and Gazebo**.  
The goal is to train an agent that can navigate a robot to a target position while **avoiding obstacles** using onboard sensors.

We compare multiple state-of-the-art RL algorithms and evaluate their performance in a simulated environment, with guidelines for real-world deployment.

---

## Algorithms Implemented

The following RL algorithms are implemented and trained:

* **SAC** – Soft Actor-Critic  
* **TD3** – Twin Delayed Deep Deterministic Policy Gradient  
* **CrossQ** – Cross Q-Learning  
* **DDPG** – Deep Deterministic Policy Gradient  
* **DQN** – Deep Q-Network  

---

## System Overview



* **Framework**: ROS2  
* **Simulator**: Gazebo  
* **Robot**: TurtleBot3  
* **Observation Space**:
    * Laser scan (LiDAR)
    * Robot odometry (Position/Orientation)
    * Relative goal position
* **Action Space**:
    * Linear velocity ($v$)
    * Angular velocity ($\omega$)

The agent receives sensor data and outputs velocity commands to reach the goal safely.

---

## Installation

You can install the project using either **Docker (recommended)** or **manual installation**.

### Option 1: Docker (Recommended)

**Prerequisites:** Docker, NVIDIA Docker (for GPU support).



```
# Build image
docker build -t smart-nav-rl .

# Run container
docker run -it --rm --net=host smart-nav-rl
```

### Option 2: Manual Installation

Requirements: Ubuntu 22.04, ROS foxy, Python 3.8


## Training

Training is done by running the robot in Gazebo and letting the RL agent interact with the environment.



1. Launch the world

```
ros2 launch turtlebot3_gazebo turtlebot3_drl_stage9.launch.py
```

2. Launch node for goal in gazebo

```
ros2 run turtlebot3_drl gazebo_goals
```

3. Launch node for drl to interact with environment

```
ros2 run turtlebot3_drl environment
```

4.Launch the drl agent 


```
ros2 run turtlebot3_drl train_agent [ALGORITHM_NAME]
```
where ALGORITHM_NAME:
    - td3
    - sac
    - ddpg
    - crossq
    - dqn


```
ros2 run turtlebot3_drl train_agent td3 
```


Load a Trained model


```
ros2 run turtlebot3_drl train_agent td3 "exmaples/td3_0_stage_9" 5000
```


## Testing 


1. Launch the world

```
ros2 launch turtlebot3_gazebo turtlebot3_drl_stage9.launch.py
```

2. Launch node for goal in gazebo

```
ros2 run turtlebot3_drl goal_test
```

3. Launch node for drl to interact with environment

```
ros2 run turtlebot3_drl environment
```

4.Launch the drl agent for testing

```
ros2 run turtlebot3_drl test_agent [ALGORITHM_NAME] [model-folder] [MODEL_EPISODE]
```

```
ros2 run turtlebot3_drl test_agent td3 "exanples/td3_0_stage_9" 5000
```

## Real World Deployment


   
