import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
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
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu6":
            activation = nn.relu6
        elif self.activation == "hardswish":
            activation = nn.hard_swish
        elif self.activation == "swish":
            activation = nn.swish
        elif self.activation == "gelu":
            activation = nn.gelu
        elif self.activation == "elu":
            activation = nn.elu
        elif self.activation == "softplus":
            activation = nn.softplus
        elif self.activation == "logsigmiod":
            activation = nn.log_sigmoid
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
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
        network = ActorCritic(env.action_space(
            env_params).n, activation=cfg.ACTIVATION)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        # Network pruning
        sparsity_distribution = functools.partial(
            jaxpruner.sparsity_distributions.uniform, sparsity=cfg.SPARSITY)
        pruner = jaxpruner.MagnitudePruning(
            sparsity_distribution_fn=sparsity_distribution,
            sparsity_type=jaxpruner.sparsity_types.NByM(64, 64))
        pruned_params, mask = pruner.instant_sparsify(
            network.init(_rng, init_x))
        network_params = pruned_params

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
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, cfg.NUM_UPDATES
        )
        return {"runner_state": runner_state, "metrics": metric}
    return train


ent_coef_search = jnp.array([0.005])  # not used

num_seeds = 10
rng = jax.random.PRNGKey(42)
rngs = jax.random.split(rng, num_seeds)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    t0 = time.time()
    activation = [
        'relu6', 'hardswish', 'swish', 'gelu',
        'elu', 'tanh', 'softplus', 'logsigmiod'
    ]
    for af in activation:
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True)
        wandb.init(project="SAFS-RL", name=af)
        cfg.ACTIVATION = af
        print(af)
        train_vvjit = jax.jit(jax.vmap(make_train(cfg)))
        outs = train_vvjit(rngs)
        print(f"time: {time.time() - t0:.2f} s")
        outs = outs["metrics"]["returned_episode_returns"]
        IQM_values = np.array([metrics.aggregate_iqm(outs[:, t])
                               for t in range(int(cfg.NUM_UPDATES))])
        mean_returns = np.zeros(int(cfg.NUM_UPDATES))
        for t in range(int(cfg.NUM_UPDATES)):
            mean_returns[t] = np.mean(outs[:, t])
        for step in range(int(cfg.NUM_UPDATES)):
            wandb.log(
                {af + " IQM": IQM_values[step], af + " Mean": mean_returns[step]}, step=step)
        wandb.finish()


if __name__ == "__main__":
    main()
