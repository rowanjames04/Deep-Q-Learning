import flappy_bird_gymnasium
import gymnasium

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample() #sample function pulls random action == 0 -> do nothing, 1 -> flap

    # Processing:
    obs, reward, terminated, _, info = env.step(action) #executes action
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()