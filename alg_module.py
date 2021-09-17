import gym
import torch.utils.data

from alg_constrants_amd_packages import *
from alg_general_functions import *
from alg_memory import ALGDataset
from alg_net import ActorNet, CriticNet
from alg_logger import run


class ALGModule:
    def __init__(
            self,
            env: gym.Env,
            critic_net_1: CriticNet,
            critic_target_net_1: CriticNet,
            critic_net_2: CriticNet,
            critic_target_net_2: CriticNet,
            actor_net: ActorNet,
            train_dataset: ALGDataset,
            train_dataloader: torch.utils.data.DataLoader
    ):
        self.env = env
        self.critic_net_1 = critic_net_1
        self.critic_target_net_1 = critic_target_net_1
        self.critic_net_2 = critic_net_2
        self.critic_target_net_2 = critic_target_net_2
        self.actor_net = actor_net
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader

        self.critic_opt_1 = torch.optim.Adam(self.critic_net_1.parameters(), lr=LR_CRITIC)
        self.critic_opt_2 = torch.optim.Adam(self.critic_net_2.parameters(), lr=LR_CRITIC)
        self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=LR_ACTOR)

        if PLOT_LIVE:
            self.fig, _ = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
            self.actor_losses = []
            self.critic_losses = []

    def fit(self):

        state = self.env.reset()

        for step in range(MAX_STEPS):
            self.validation_step(step)

            action = get_action(state, self.actor_net, step)
            next_state, reward, done, _ = self.env.step(action)

            experience = Experience(state=state, action=action, reward=reward, done=done, new_state=next_state)
            self.train_dataset.append(experience)

            state = next_state if done else self.env.reset()

            self.training_step(step)

        self.env.close()

    def training_step(self, step):
        if step % UPDATE_EVERY == 0:
            print(f'[TRAINING STEP] Step: {step}')

            list_of_batches = list(self.train_dataloader)
            n_batches_to_iterate = min(len(list_of_batches), BATCHES_IN_TRAINING_STEP)

            for b_indx in range(n_batches_to_iterate):  # range(len(list_of_batches)) | range(x)
                batch = list_of_batches[b_indx]
                states, actions, rewards, dones, next_states = batch

                # compute targets
                actions_target_net = self.actor_net(next_states)
                Q_target_vals_1 = self.critic_target_net_1(next_states, actions_target_net)
                Q_target_vals_2 = self.critic_target_net_2(next_states, actions_target_net)
                min_Q_vals = torch.minimum(Q_target_vals_1, Q_target_vals_2)
                log_policy = torch.log()
                y = rewards.float() + GAMMA * (~dones).float() * torch.squeeze(Q_target_vals)

                # update critic - gradient descent
                self.critic_opt.zero_grad()
                actions_net = self.actor_net(states)
                Q_vals = self.critic_net(states, actions_net)
                Q_vals = torch.squeeze(Q_vals)
                critic_loss = nn.MSELoss()(Q_vals, y.detach())
                critic_loss.backward()
                self.critic_opt.step()

                # update actor - gradient ascent
                self.actor_opt.zero_grad()
                actions_net = self.actor_net(states)
                actor_loss = - self.critic_net(states, actions_net).mean()
                actor_loss.backward()
                self.actor_opt.step()

                # update target networks
                critic_w_mse, actor_w_mse = [], []
                for target_param, param in zip(self.critic_target_net.parameters(), self.critic_net.parameters()):
                    target_param.data.copy_(POLYAK * target_param.data + (1.0 - POLYAK) * param.data)
                    critic_w_mse.append(np.square(target_param.data.numpy() - param.data.numpy()).mean())

                for target_param, param in zip(self.actor_target_net.parameters(), self.actor_net.parameters()):
                    target_param.data.copy_(POLYAK * target_param.data + (1.0 - POLYAK) * param.data)
                    actor_w_mse.append((np.square(target_param.data.numpy() - param.data.numpy())).mean())

                self.plot(
                    {
                        'actor_loss': actor_loss.item(),
                        'critic_loss': critic_loss.item(),
                        'critic_w_mse': critic_w_mse,
                        'actor_w_mse': actor_w_mse,
                        'b_indx': b_indx,
                    }
                    # {'rewards': rewards, 'values': values, 'ref_v': ref_v.numpy(),
                    # 'loss': self.log_for_loss, 'lengths': lengths, 'adv_v': adv_v.numpy()}
                )
                self.neptune_update(loss=None)

    def validation_step(self, step):
        if step % VAL_CHECKPOINT_INTERVAL == 0 and step > 0:
            print(f'[VALIDATION STEP] Step: {step}')
            play(1, self.actor_net)

    def configure_optimizers(self):
        pass

    def plot(self, graph_dict):
        # plot live:
        if PLOT_LIVE:
            def plot_graph(ax, indx, list_of_values, label, color='r'):
                ax[indx].cla()
                ax[indx].plot(list(range(len(list_of_values))), list_of_values, c=color)  # , edgecolor='b')
                ax[indx].set_title(f'Plot: {label}')
                ax[indx].set_xlabel('iters')
                ax[indx].set_ylabel(f'{label}')

            ax = self.fig.get_axes()
            b_indx = graph_dict['b_indx']

            self.actor_losses.append(graph_dict['actor_loss'])
            self.critic_losses.append(graph_dict['critic_loss'])

            # graphs
            if b_indx % 9 == 0:
                plot_graph(ax, 1, self.actor_losses, 'actor_loss')
                plot_graph(ax, 2, self.critic_losses, 'critic_loss')
                plot_graph(ax, 3, graph_dict['actor_w_mse'], 'actor_w_mse')
                plot_graph(ax, 4, graph_dict['critic_w_mse'], 'critic_w_mse')

                plt.pause(0.05)
                # plt.pause(1.05)

    @staticmethod
    def neptune_update(loss):
        if NEPTUNE:
            run['acc_loss'].log(loss)
            run['acc_loss_log'].log(f'{loss}')







