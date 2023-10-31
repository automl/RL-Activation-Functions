import flax.linen as nn
import numpy as np
import optax
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import time
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import jaxpruner
import functools
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation_af: str = "elu"
    critic_af: str = "elu"
    activation_list: str = "elu"

    def set_activation(self, new_activation):
        if (isinstance(new_activation, str)):
            self.activation = new_activation
            self.critic = new_activation
        else:
            self.activation_list = new_activation

    @nn.compact
    def __call__(self, x):
        def activation_fucntion(activation_function):
            if activation_function == "relu6":
                af = nn.relu6
            elif activation_function == "hardswish":
                af = nn.hard_swish
            elif activation_function == "swish":
                af = nn.swish
            elif activation_function == "gelu":
                af = nn.gelu
            elif activation_function == "elu":
                af = nn.elu
            elif activation_function == "softplus":
                af = nn.softplus
            elif activation_function == "logsigmoid":
                af = nn.log_sigmoid
            elif activation_function == "tanh":
                af = nn.tanh
            else:
                af = nn.tanh
            return af
        activation_list = self.activation_list.split(", ")
        if len(activation_list) == 4:
            af_policy_1 = activation_fucntion(activation_list[0])
            af_policy_2 = activation_fucntion(activation_list[1])
            af_critic_1 = activation_fucntion(activation_list[2])
            af_critic_2 = activation_fucntion(activation_list[3])
        else:
            af_policy_1 = activation_fucntion(self.activation_af)
            af_policy_2 = af_policy_1
            af_critic_1 = activation_fucntion(self.critic_af)
            af_critic_2 = af_critic_1
        print("policy 1:", activation_list[0], " policy 2:", activation_list[1],
              " critic 1:", activation_list[2], " critic 2:", activation_list[3])
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = af_policy_1(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = af_policy_2(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = af_critic_1(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = af_critic_2(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(cfg):
    cfg.NUM_UPDATES = cfg.TOTAL_TIMESTEPS // cfg.NUM_STEPS // cfg.NUM_ENVS
    cfg.MINIBATCH_SIZE = cfg.NUM_ENVS * cfg.NUM_STEPS // cfg.NUM_MINIBATCHES

    env, env_params = gymnax.make(cfg.ENV_NAME)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (cfg.NUM_MINIBATCHES
                      * cfg.UPDATE_EPOCHS) / cfg.NUM_UPDATES)
        return cfg.LR * frac

    def train(rng):
        # INIT NETWORK
        # ent_coef = jnp.array([0.01, 0.005, 0.001])
        ent_coef = cfg.ENT_COEF
        if cfg.LAYERS == True:
            network = ActorCritic(env.action_space(
                env_params).n, activation_list=cfg.ACTIVATION_LIST)
        else:
            network = ActorCritic(env.action_space(
                env_params).n, activation_af=cfg.ACTIVATION, critic_af=cfg.CRITIC_ACTIVATION)

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        # Network pruning
        # sparsity_distribution = functools.partial(
        #     jaxpruner.sparsity_distributions.uniform, sparsity=cfg.SPARSITY)
        # pruner = jaxpruner.MagnitudePruning(
        #     sparsity_distribution_fn=sparsity_distribution,
        #     sparsity_type=jaxpruner.sparsity_types.NByM(64, 64))
        # pruned_params, mask = pruner.instant_sparsify(
        #     network.init(_rng, init_x))
        network_params = network.init(_rng, init_x)
        if cfg.ANNEAL_LR:
            tx = optax.chain(
                optax.clip_by_global_norm(cfg.MAX_GRAD_NORM),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(
                cfg.MAX_GRAD_NORM), optax.adam(cfg.LR, eps=1e-5))
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, cfg.NUM_ENVS)
        obsv, env_state = jax.vmap(
            env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, cfg.NUM_ENVS)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, cfg.NUM_STEPS
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + cfg.GAMMA * \
                        next_value * (1 - done) - value
                    gae = (
                        delta
                        + cfg.GAMMA *
                        cfg.GAE_LAMBDA * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-cfg.CLIP_EPS, cfg.CLIP_EPS)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses,
                                              value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - cfg.CLIP_EPS,
                                1.0 + cfg.CLIP_EPS,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + cfg.VF_COEF * value_loss
                            - ent_coef * entropy  # check ent_coef
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = cfg.MINIBATCH_SIZE * cfg.NUM_MINIBATCHES
                assert (
                    batch_size == cfg.NUM_STEPS * cfg.NUM_ENVS
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [cfg.NUM_MINIBATCHES, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch,
                                advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, cfg.UPDATE_EPOCHS
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        interval = round(cfg.NUM_UPDATES / 2)
        runner_state, metric_1 = jax.lax.scan(
            _update_step, runner_state, None, interval
        )
        # network.set_activation("elu")
        # train_state = TrainState.replace(
        #    train_state, apply_fn=network.apply, params=runner_state[0].params)
        # runner_state = (train_state, runner_state[1],
        #                runner_state[2], runner_state[3])
        runner_state, metric_2 = jax.lax.scan(
            _update_step, runner_state, None, cfg.NUM_UPDATES - interval
        )
        metric = {}
        for entry in metric_1:
            metric[entry] = jnp.concatenate(
                (metric_1[entry], metric_2[entry]), axis=0)
        # runner_state, metric = jax.lax.scan(
        #    _update_step, runner_state, None, cfg.NUM_UPDATES
        # )
        return {"runner_state": runner_state, "metrics": metric}
    return train


num_seeds = 10
rng = jax.random.PRNGKey(42)
rngs = jax.random.split(rng, num_seeds)
window_size = 10000


def moving_avg(arr):
    num_seeds, num_updates, num_steps, num_envs = arr.shape
    moving_average = np.zeros((
        num_seeds, num_updates * num_steps * num_envs - window_size + 1), dtype=float)
    for seeds in range(num_seeds):
        data = arr[seeds].reshape(-1)
        moving_average_values = np.convolve(
            data, np.ones(window_size) / window_size, mode='valid')
        moving_average[seeds] = moving_average_values
    return moving_average


@hydra.main(version_base=None, config_path="configs", config_name="config_cartpole")
def main(cfg: DictConfig):
    t0 = time.time()
    label = ""
    path = "{env}_Plots/".format(env=cfg.ENV_NAME)
    if (cfg.LAYER == True):
        activation_list = cfg.ACTIVATION_LIST.split(", ")
    else:
        activation_list = cfg.ACTIVATION_FUNCTIONS.split(", ")
    activation = cfg.ACTIVATION_FUNCTIONS.split(", ")
    # activation_list = ["tanh"]
    # activation = ["tanh"]
    for j, af_critic in enumerate(activation_list):
        IQM_values_list = []
        for af_policy in activation:
            cfg.ACTIVATION = af_policy
            cfg.CRITIC_ACTIVATION = af_critic
            if (cfg.LAYER == True):
                af_layer = activation_list
                af_layer[j] = af_policy
                cfg.ACTIVATION_LIST = ", ".join(af_layer)
                label = "Layer: {count}, AF:{af}".format(
                    count=j+1, af=af_policy)
                print("Layer: ", j+1, "AF: ", af_policy)

            else:
                label = "Policy:{policy}".format(
                    policy=af_policy)
            train_vvjit = jax.jit(jax.vmap(make_train(cfg)))
            outs = train_vvjit(rngs)
            outs = outs["metrics"]["returned_episode_returns"]
            outs = moving_avg(outs)
            print(f"time: {time.time() - t0:.2f} s")
            # average_values = moving_avg(outs)
            # num_steps = average_values.shape[1]
            # IQM_values = np.array([metrics.aggregate_iqm(average_values[:, t])
            #                       for t in range(num_steps)])
            # IQM_values_list.append(IQM_values)
            IQM_values_list.append(outs)
            print(f"time: {time.time() - t0:.2f} s")

        if (cfg.LAYER == True and cfg.ENT_COEF != 0):
            label = "{env}_entCoef_Critic_{af}_Layer{num}.npy".format(env=cfg.ENV_NAME,
                                                                      policy=af_policy, af=af_critic, num=j+1)
        elif (cfg.LAYER == True):
            label = "{env}_Critic_{af}_Layer{num}.npy".format(env=cfg.ENV_NAME,
                                                              policy=af_policy, af=af_critic, num=j+1)
        elif (cfg.ENT_COEF != 0):
            label = "{env}_entCoef_Critic_{af}".format(
                env=cfg.ENV_NAME, af=af_critic)
        else:
            label = "{env}_Critic_{af}".format(env=cfg.ENV_NAME, af=af_critic)
        np.save(path + "numpy/" + label, IQM_values_list)


if __name__ == "__main__":
    main()
