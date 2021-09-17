import numpy as np

from alg_constrants_amd_packages import *


def get_action(state, model: nn.Module, step=0):
    with torch.no_grad():
        model_output = model(np.expand_dims(state, axis=0))
        model_output = torch.squeeze(model_output)
        action = model_output.detach().numpy()
        noise = ACT_NOISE / np.log(step) if step > 5000 else ACT_NOISE
        action = action + np.random.normal(0, noise, 2)
        action = np.clip(action, -1, 1)
        return action


def fill_the_buffer(train_dataset, env, actor_net):
    state = env.reset()
    while len(train_dataset) < UPDATE_AFTER:
        action = get_action(state, actor_net)
        next_state, reward, done, _ = env.step(action)
        # env.render()
        experience = Experience(state=state, action=action, reward=reward, done=done, new_state=next_state)
        train_dataset.append(experience)
        if done:
            state = env.reset()

    env.close()


def play(times: int = 1, model: nn.Module = None):
    with torch.no_grad():
        env = gym.make(ENV)
        state = env.reset()
        game = 0
        total_reward = 0
        while game < times:
            if model:
                action = get_action(state, model)
            else:
                action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
            if done:
                state = env.reset()
                game += 1
                print(f'finished game {game} with a total reward: {total_reward}')
                total_reward = 0
            else:
                state = next_state
        env.close()




