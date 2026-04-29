from __future__ import annotations

from collections.abc import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random as jr
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from Algorithms.jax_support import ActorCritic
from Algorithms.td3_jax.config import TD3Config
from Algorithms.jax_support import compute_and_apply_gradients
from Algorithms.jax_support import QFunctionEnsemble
from Algorithms.jax_support import Policy
from Algorithms.jax_support import register
from Algorithms.jax_support import get_optimizer

@register("td3", TD3Config)
class TD3Agent(ActorCritic):
    actor: Policy
    critic: QFunctionEnsemble
    target_actor: Policy
    target_critic: QFunctionEnsemble

    step: Array

    policy_noise: float = eqx.field(static=True)
    target_noise: float = eqx.field(static=True)
    noise_clip: float = eqx.field(static=True)
    policy_delay: int = eqx.field(static=True)
    
    action_space_low: Array
    action_space_high: Array

    gamma: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)

    @classmethod
    def _create_kwargs(
        cls, config: TD3Config, *, d_obs: int, d_act: int, key: PRNGKeyArray
    ) -> dict:
        f = cls._base_fields(config, d_obs=d_obs, d_act=d_act, key=key)
        
        low = jnp.array(config.action_space_low)
        high = jnp.array(config.action_space_high)
        
        return {
            "actor": f["actor"],
            "critic": f["critic"],
            "target_actor": f["actor"],
            "target_critic": f["target_critic"],
            "actor_optimizer": f["actor_optimizer"],
            "critic_optimizer": f["critic_optimizer"],
            "actor_opt_state": f["actor_opt_state"],
            "critic_opt_state": f["critic_opt_state"],
            "step": jnp.array(0, dtype=jnp.int32),
            "policy_noise": config.policy_noise,
            "target_noise": config.target_noise,
            "noise_clip": config.noise_clip,
            "policy_delay": config.policy_delay,
            "action_space_low": low,
            "action_space_high": high,

            "gamma": config.gamma,
            "tau": config.tau,
        }
    
    @eqx.filter_jit
    def eval_act(self, observation: Float[Array, "obs_dim"]) -> Float[Array, "act_dim"]:
        action = self.actor.net(observation)
        return jnp.clip(action, self.action_space_low, self.action_space_high)
    
    @eqx.filter_jit
    def exploration_act(self, observation: Float[Array, "obs_dim"], key: PRNGKeyArray) -> Float[Array, "act_dim"]:
        action = self.actor.net(observation)
        noise = jr.normal(key, shape=action.shape) * self.policy_noise
        action = action + noise
        return jnp.clip(action, self.action_space_low, self.action_space_high)
    
    def actor_loss(self, actor: Policy, batch: dict[str, Array], key: PRNGKeyArray) -> tuple[Scalar, dict]:
        return compute_td3_actor_loss(actor, batch["observation"], self.critic)
    
    def critic_loss(self, critic: QFunctionEnsemble, batch: dict[str, Array], key: PRNGKeyArray) -> tuple[Scalar, dict]:
        return compute_td3_critic_loss(critic, self.target_critic, 
                                            self.target_actor, batch, self.gamma, self.target_noise, 
                                            self.noise_clip, self.action_space_low, self.action_space_high, key = key)

    def _td3_update(self, batch: dict[str, Array], key: PRNGKeyArray) -> tuple["TD3", dict]:
        critic_key, actor_key = jr.split(key)
        new_step = self.step + 1

        new_critic, new_critic_opt_state, critic_loss_val, critic_aux = compute_and_apply_gradients(
            self.critic,
            lambda c: self.critic_loss(c, batch, critic_key),
            self.critic_optimizer,
            self.critic_opt_state,
        )

        should_update = new_step % self.policy_delay == 0

        def perform_updates():
            new_actor, new_actor_opt, actor_loss, actor_aux = compute_and_apply_gradients(
                self.actor,
                lambda a: self.actor_loss(a, batch, actor_key),
                self.actor_optimizer,
                self.actor_opt_state,
            )
            n_target_actor = _soft_update(self.target_actor, new_actor, self.tau)
            n_target_critic = _soft_update(self.target_critic, new_critic, self.tau)
            return new_actor, new_actor_opt, n_target_actor, n_target_critic, actor_loss, actor_aux
        
        def skip_updates():
            return (self.actor, self.actor_opt_state, self.target_actor, self.target_critic, jnp.array(0.0), {"actor_q_values": jnp.array(0.0)})

        new_actor, new_actor_opt_state, new_target_actor, new_target_critic, actor_loss_val, actor_aux = jax.lax.cond(
            should_update, perform_updates, skip_updates
        )

        new_self = eqx.tree_at(
            lambda m: (m.critic, m.critic_opt_state, m.actor, m.actor_opt_state, m.target_actor, m.target_critic, m.step),
            self,
            (new_critic, new_critic_opt_state, new_actor, new_actor_opt_state, new_target_actor, new_target_critic, new_step),
        )

        return new_self, {"critic_loss": critic_loss_val, "actor_loss": actor_loss_val, **critic_aux, **actor_aux}
    
    @eqx.filter_jit
    def learn(self, batch: dict[str, Array], *, key: PRNGKeyArray) -> tuple["TD3", dict]:
        return self._td3_update(batch, key)
    
class TD3:
    def __init__(self, env):
        self.key = jr.PRNGKey(0)

        d_obs = env.observation_space.shape[0]
        d_act = env.action_space.shape[0]

        self.action_low = jnp.array(env.action_space.low)
        self.action_high = jnp.array(env.action_space.high)
        
        config = TD3Config()

        kwargs = TD3Agent._create_kwargs(config, d_obs=d_obs, d_act=d_act, key=self.key)
        self.agent = TD3Agent(**kwargs)
        
        self.buffer = []

    def select_action(self, state, evaluate=False):
        state_jnp = jnp.array(state)
        self.key, subkey = jr.split(self.key)

        action = self.agent.actor.net(state_jnp)

        if not evaluate:
            action = action + jr.normal(subkey, shape=action.shape) * 0.1
            
        action = jnp.clip(action, self.action_low, self.action_high)
        return jnp.array(action)

    def step(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

        if len(self.buffer) >= 1000:
            pass

    def save_checkpoint(self, path, ep, eval_rewards):
        eqx.tree_serialise_leaves(path, self.agent)

    def load_checkpoint(self, path):
        # Dummy für Kompatibilität
        return {}

def _soft_update(target, source, tau):
    return jax.tree_util.tree_map(lambda t, s: (1 - tau) * t + tau * s, target, source)

def compute_td3_actor_loss(
        actor: Policy, obs: Float[Array, "batch obs_dim"], critic: QFunctionEnsemble
    ) -> tuple[Scalar, dict]:
    actions = jax.vmap(actor.net)(obs)
    q_values = jax.vmap(lambda o, a: critic(o, a)[0])(obs, actions)
    actor_loss = -jnp.mean(q_values)
    return actor_loss, {"actor_q_values": jnp.mean(q_values)}

def compute_td3_critic_loss(
    critic: QFunctionEnsemble,
    target_critic: QFunctionEnsemble,
    target_actor: Policy,
    batch: dict[str, Array],
    gamma: float,
    target_policy_noise: float,
    target_policy_noise_clip: float,
    low: Array,
    high: Array,
    key: PRNGKeyArray,
) -> tuple[Scalar, dict]:
    
    next_obs = batch["next_observation"]

    next_actions = jax.vmap(target_actor.net)(next_obs)
    noise = jr.normal(key, shape=next_actions.shape) * target_policy_noise
    noise = jnp.clip(noise, -target_policy_noise_clip, target_policy_noise_clip)
    next_actions_smoothed = jnp.clip(next_actions + noise, low, high)

    target_q_ensemble = jax.vmap(target_critic)(next_obs, next_actions_smoothed)
    target_q_min = target_q_ensemble.min(axis=-1)
    target_q = batch["reward"] + gamma * (1 - batch["terminated"]) * target_q_min

    current_q_ensemble = jax.vmap(critic)(batch["observation"], batch["action"])
    loss = jnp.mean(jnp.square(current_q_ensemble - jnp.expand_dims(target_q, axis=-1)).sum(axis=-1))
    return loss, {}