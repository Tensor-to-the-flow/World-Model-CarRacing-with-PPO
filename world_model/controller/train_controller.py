from .ppo import PPO
from ..data.car_racing import CarRacingWrapper


def train_controller(model, env, max_episodes=1000):

    for ep in range(max_episodes):
        state_batch = []
        action_batch = []
        reward_batch = []
        old_policy_batch = []

        episode_reward, done = 0, False

        state = env.reset()
        env.viewer.window.dispatch_events()

        while not done:
            log_old_policy, action = model.actor.get_action(state)

            next_state, reward, done, _ = self.env.step(action)

            state = np.reshape(state, [1, self.state_dim])
            action = np.reshape(action, [1, self.action_dim])
            next_state = np.reshape(next_state, [1, self.state_dim])
            reward = np.reshape(reward, [1, 1])
            log_old_policy = np.reshape(log_old_policy, [1, 1])

            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append((reward + 8) / 8)
            old_policy_batch.append(log_old_policy)

            if len(state_batch) >= args.update_interval or done:
                states = self.list_to_batch(state_batch)
                actions = self.list_to_batch(action_batch)
                rewards = self.list_to_batch(reward_batch)
                old_policys = self.list_to_batch(old_policy_batch)

                v_values = self.critic.model.predict(states)
                next_v_value = self.critic.model.predict(next_state)

                gaes, td_targets = self.gae_target(
                    rewards, v_values, next_v_value, done)

                for epoch in range(args.epochs):
                    actor_loss = self.actor.train(
                        old_policys, states, actions, gaes)
                    critic_loss = self.critic.train(states, td_targets)

                state_batch = []
                action_batch = []
                reward_batch = []
                old_policy_batch = []

            episode_reward += reward[0][0]
            state = next_state[0]

        print('EP{} EpisodeReward={}'.format(ep, episode_reward))
        # wandb.log({'Reward': episode_reward})


if __name__ == "__main__":

    carracing = CarRacingWrapper()
    model = PPO(carracing.action_space.high[0], )

    train_controller(model, carracing)
