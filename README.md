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

1. build docker image

```
# Build image
bash docker-build.bash
```

2. run docker image
```
# Run container
bash docker-run.bash
```

3. another terminal for same container

```
# Run same  container again
bash into_docker.bash
```

### Option 2: Manual Installation

Requirements: Ubuntu 22.04, ROS foxy, Python 3.8

## **Installing ROS2**
Install ROS2 foxy according to the following guide: [link](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html). You can choose either the Desktop or Bare Bones ROS installation, both work. <br>
To prevent having to manually source the setup script every time, add the following line at the end of your `~/.bashrc` file:

```
source /opt/ros/foxy/setup.bash
```

More detailed installation instructions can be found [here](https://automaticaddison.com/how-to-install-ros-2-foxy-fitzroy-on-ubuntu-linux/).


## **Installing Gazebo**

For this project we will be using Gazebo **11.0.** To install Gazebo 11.0, navigate to the following [page](http://gazebosim.org/tutorials?tut=install_ubuntu), select Version 11.0 in the top-right corner and follow the default installation instructions.

Next, we need to install a package that allows ROS2 to interface with Gazebo.
To install this package we simply execute the following command in a terminal:
```
sudo apt install ros-foxy-gazebo-ros-pkgs
```
After successful installation we are now going to test our ROS2 + Gazebo setup by making a demo model move in the simulator. First, install two additional packages for demo purposes (they might already be installed):
```
sudo apt install ros-foxy-ros-core ros-foxy-geometry2
```
Source ROS2 before we launch the demo:
```
source /opt/ros/foxy/setup.bash
```

Now, let's load the demo model in gazebo:
```
gazebo --verbose /opt/ros/foxy/share/gazebo_plugins/worlds/gazebo_ros_diff_drive_demo.world
```
This should launch the Gazebo GUI with a simple vehicle model. Open a second terminal and provide the following command to make the vehicle move:
```
ros2 topic pub /demo/cmd_demo geometry_msgs/Twist '{linear: {x: 1.0}}' -1
```
If the vehicle starts moving forward we confirmed that the Gazebo-ROS connection works.
If something does not work, carefully check whether you executed all the commands and sourced ROS2 (`source /opt/ros/foxy/setup.bash`). You can also check the more detailed [guide](https://automaticaddison.com/how-to-install-ros-2-foxy-fitzroy-on-ubuntu-linux/).

## **Installing Python3, Pytorch**

If you are using Ubuntu 20.04 as specified, Python should already be preinstalled. The last tested version for this project was Python 3.8.10

Install pip3 (python package manager for python 3) as follows:
```
sudo apt install python3-pip
```

To install the tested version of PyTorch (1.10.0) with CUDA support (11.3) and packages for generating graphs, run:
```
pip3 install matplotlib pandas pyqtgraph==0.12.4 PyQt5==5.14.1 torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

`pyqtgraph` and `PyQt` are optional and only necessary if you want to visualize the neural network activity. `pandas` is only required for generating graphs outside of training.

**Note: The version of CUDA support to install will depend on the [compute capability](https://developer.nvidia.com/cuda-gpus) of your GPU**

## **Enabling GPU support (recommended)**

We can significantly speed up the training procedure by making use of a GPU. If no GPU is available or it is not initialized correctly the training will automatically be redirected to the CPU. Since most users have access to an NVIDIA GPU we will explain how to enable this to work with PyTorch on linux.
Three different components are required to train on GPU:
- NVIDIA drivers for linux
- The CUDA library for linux
- cuDNN (comes with pytorch and should be installed automatically)

Press the windows/command key and type "Additional drivers" to make the corresponding linux menu come up. Here, multiple radio button options should be listed for installing different nvidia drivers. Install the option with the latest version (highest number, e.g. currently nvidia-driver-510).

The next step is to download the correct CUDA version. This will depend on your NVIDIA drivers and GPU variant. Generally, all you have to do is execute:
```
sudo apt install nvidia-cuda-toolkit
```
You can then verify that CUDA is installed using:
```
nvcc -V
```
and
```
nvidia-smi
```
Which should display version numbers and GPU information.
In case of doubt, consult one of the following resources: [one](https://varhowto.com/install-pytorch-ubuntu-20-04/), [two](https://pytorch.org/get-started/locally/), [three](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)

## **Downloading the code base and building**
<!-- Now it's time to create a workspace that will serve as the basis for our project. To do this, follow the tutorial [here](https://automaticaddison.com/how-to-create-a-workspace-ros-2-foxy-fitzroy/) -->

Now it's time to download the repository to the actual code.

Since ROS2 does not yet support metapackages, we will have to download the whole workspace from Git.

First, make sure you have the `turtlebot3-description` package by running:
```
sudo apt-get install ros-foxy-turtlebot3-description
```

Open a terminal in the desired location for the new workspace. Clone the repository using:
```
git clone https://github.com/CharanjitBK/Reinforcement-Learning-Based-Smart-Navigation-for-TurtleBot.git
```

`cd` into the directory and make sure you are on the main branch
```
cd turtlebot3_drlnav
git checkout main
```

Next, install the correct rosdep tool
```
sudo apt install python3-rosdep2
```

Then initialize rosdep by running
```
rosdep update
```

Now we can use rosdep to install all ROS packages needed by our repository
```
rosdep install -i --from-path src --rosdistro foxy -y
```

Now that we have all of the packages in place it is time to build the repository. First update your package list
```
sudo apt update
```

Then install the build tool **colcon** which we will use to build our ROS2 package
```
sudo apt install python3-colcon-common-extensions
```

Next, it's time to actually build the repository code!
```
colcon build
```
After colcon has finished building source the repository
```
source install/setup.bash
```

The last thing we need to do before running the code is add a few lines to our `~/.bashrc` file so that they are automatically executed whenever we open a new terminal. Add the following lines at the end of your `~/.bashrc` file and **replace ~/path/to/turtlebot3_drlnav/repo with the path where you cloned the repository. (e.g. ~/turtlebot3_drlnav)**
```
# ROS2 domain id for network communication, machines with the same ID will receive each others' messages
export ROS_DOMAIN_ID=1

# Fill in the path to where you cloned the turtlebot3_drlnav repo
WORKSPACE_DIR=~/path/to/turtlebot3_drlnav
export DRLNAV_BASE_PATH=$WORKSPACE_DIR

# Source the workspace
source $WORKSPACE_DIR/install/setup.bash

# Allow gazebo to find our turtlebot3 models
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$WORKSPACE_DIR/src/turtlebot3_simulations/turtlebot3_gazebo/models

# Select which turtlebot3 model we will be using (default: burger, waffle, waffle_pi)
export TURTLEBOT3_MODEL=burger

# Allow Gazebo to find the plugin for moving the obstacles
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:$WORKSPACE_DIR/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_drl_world/obstacle_plugin/lib
```

For more detailed instructions on ros workspaces check [this guide](https://automaticaddison.com/how-to-create-a-workspace-ros-2-foxy-fitzroy/).

**Note: Always make sure to first run ```source install/setup.bash``` or open a fresh terminal after building with `colcon build`.**

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

#### Download Pre-trained Model

The model weights are hosted on Google Drive.  
Run the script to download and extract:

```bash
python download_model.py
```

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


connect with the turtlebot

```
ros2 launch turtlebot3_bringup robot.launch.py
```

1. launch real environment node for  turtlebot to interact with environment

```
ros2 run turtlebot3_drl real_environment
```

2.  Launch node for goal

```
ros2 run turtlebot3_drl goal_Real
```

3.Launch the drl agent in real world

```
ros2 run turtlebot3_drl real_agent [ALGORITHM_NAME] [model-folder] [MODEL_EPISODE]
```

```
ros2 run turtlebot3_drl real_agent td3 "exanples/td3_0_stage_9" 5000
```




   
