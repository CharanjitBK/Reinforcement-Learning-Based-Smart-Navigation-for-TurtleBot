from ..common.settings import REWARD_FUNCTION, COLLISION_OBSTACLE, COLLISION_WALL, TUMBLE, SUCCESS, TIMEOUT, RESULTS_NUM

goal_dist_initial = 0

reward_function_internal = None

def get_reward(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance):
    return reward_function_internal(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance)

def get_reward_A(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # [-3.14, 0]
        r_yaw = -1 * abs(goal_angle)

        # [-4, 0]
        r_vangular = -1 * (action_angular**2)

        # [-1, 1]
        r_distance = (2 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 1

        # [-20, 0]
        if min_obstacle_dist < 0.22:
            r_obstacle = -20
        else:
            r_obstacle = 0

        # [-2 * (2.2^2), 0]
        r_vlinear = -1 * (((0.22 - action_linear) * 10) ** 2)

        reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular - 1

        if succeed == SUCCESS:
            reward += 2500
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
            reward -= 2000
        return float(reward)

def get_reward_B(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
    # 1. Orientation Reward (Encourage facing the goal)
    # Scaled to [-1, 0]
    r_yaw = -1 * (abs(goal_angle) / 3.14)

    # 2. Progress Reward (The "Breadcrumb")
    # This is much better than your current r_distance formula.
    # If the robot is closer than the last step, it gets a positive reward.
    # You need to pass 'last_goal_dist' into this function to do this perfectly.
    # Otherwise, use a simple distance-based reward:
    r_distance = 0.5 * (goal_dist_initial - goal_dist) / goal_dist_initial

    # 3. Smoothness Penalty (Slightly discourage excessive spinning)
    r_vangular = -0.1 * (action_angular**2)

    # 4. Obstacle Avoidance (Shaped penalty)
    # Instead of a hard -20, give a "warning" as it gets closer
    if min_obstacle_dist < 0.25:
        r_obstacle = -2.0 * (0.25 - min_obstacle_dist)
    else:
        r_obstacle = 0

    # 5. Living Penalty (Small)
    # This encourages the robot to reach the goal FAST.
    r_living = -0.1 

    reward = r_yaw + r_distance + r_obstacle + r_vangular + r_living

    # 6. Terminal Rewards (Keep these high but balanced)
    if succeed == SUCCESS:
        reward += 500.0  # Reduced from 2500 to keep scales manageable for PPO
    elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
        reward -= 500.0  # Match the success scale
    
    return float(reward)
# Define your own reward function by defining a new function: 'get_reward_X'
# Replace X with your reward function name and configure it in settings.py

def reward_initalize(init_distance_to_goal):
    global goal_dist_initial
    goal_dist_initial = init_distance_to_goal

function_name = "get_reward_" + REWARD_FUNCTION
reward_function_internal = globals()[function_name]
if reward_function_internal == None:
    quit(f"Error: reward function {function_name} does not exist")
