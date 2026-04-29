from typing import Literal

from abc import abstractmethod
from collections.abc import Callable

import distrax
import equinox as eqx
import optax
import jax
from jax import random as jr
from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from dataclasses import dataclass, field

class ValueFunction(eqx.Module):
    """Abstract base class for value functions."""

    net: eqx.Module

class Ensemble(eqx.Module):
    n_ensemble: int  # Nr of ensemble members
    ensemble: eqx.Module  # the ensemble

    def __init__(self, net_fun: Callable, n_ensemble: int, key: PRNGKeyArray):
        self.n_ensemble = n_ensemble
        keys = jr.split(key, n_ensemble)
        self.ensemble = eqx.filter_vmap(net_fun)(keys)

    def _get_member(self, idx: int) -> eqx.Module:
        arrays, static = eqx.partition(self.ensemble, eqx.is_array)
        indexed_arrays = jax.tree.map(lambda x: x[idx], arrays)
        return eqx.combine(indexed_arrays, static)

    def __getitem__(self, idx: int) -> eqx.Module:
        """Extract a single member from the ensemble."""
        return self._get_member(idx)

    def __call__(self, x: Array) -> Array:
        """Apply the ensemble forward to a single input."""
        return eqx.filter_vmap(lambda net, x: net(x), in_axes=(eqx.if_array(0), None))(
            self.ensemble, x
        )

    def __len__(self) -> int:
        """Return the number of members in the ensemble."""
        return self.n_ensemble

class QFunctionEnsemble(ValueFunction):
    """Abstract base class for Q-function ensembles."""

    net: Ensemble

    def __call__(
        self, observation: Float[Array, "obs_dim"], action: Float[Array, "act_dim"]
    ) -> Float[Array, "n_ensemble"]:
        x = jnp.concatenate([observation, action], axis=-1)

        return self.net(x).squeeze(-1)  # assume scalar output space

class Agent(eqx.Module):
    @abstractmethod
    def act(
        self,
        obs: Float[Array, "obs_dim"],
        key: PRNGKeyArray,
    ) -> Float[Array, "act_dim"]:
        raise NotImplementedError

    @abstractmethod
    def learn(
        self,
        batch: dict[str, Array],
        key: PRNGKeyArray,
    ) -> tuple["Agent", dict]:
        raise NotImplementedError

    def save_state(self) -> None:
        raise NotImplementedError

@dataclass
class NetworkEntry:
    hidden_dim: int = 256
    depth: int = 2
    n_ensemble: int = 1
    has_target: bool = False
    d_out: int | None = None

@dataclass
class BaseNetworkConfig:
    actor: NetworkEntry = field(default_factory=NetworkEntry)
    critic: NetworkEntry = field(
        default_factory=lambda: NetworkEntry(n_ensemble=2, has_target=True)
    )

@dataclass
class OptimizerEntry:
    lr: float = 3e-4
    optim: Literal["adam", "adamw"] = "adam"
    max_grad_norm: float | None = None


@dataclass
class BaseOptimizerConfig:
    actor: OptimizerEntry = field(default_factory=OptimizerEntry)
    critic: OptimizerEntry = field(default_factory=OptimizerEntry)

@dataclass
class BaseModelConfig:
    gamma: float = 0.99
    tau: float = 0.005
    network: BaseNetworkConfig = field(default_factory=BaseNetworkConfig)
    optimizer: BaseOptimizerConfig = field(default_factory=BaseOptimizerConfig)

class Policy(eqx.Module):
    net: eqx.Module

class StochasticPolicy(Policy):
    @abstractmethod
    def __call__(
        self, observation: Float[Array, "obs_dim"], *, key: PRNGKeyArray
    ) -> Float[Array, "act_dim"]:
        raise NotImplementedError

    def mean_action(
        self, observation: Float[Array, "obs_dim"]
    ) -> Float[Array, "act_dim"]:
        raise NotImplementedError

    @abstractmethod
    def sample_and_log_prob(
        self, observation: Float[Array, "obs_dim"], *, key: PRNGKeyArray
    ) -> tuple[Float[Array, "act_dim"], Float[Array, ""]]:
        raise NotImplementedError


# class TanhGaussianPolicy(StochasticPolicy):
    """Tanh-Gaussian policy: tanh(N(mu, sigma))."""

    log_std_min: float = eqx.field(static=True, default=-20.0)
    log_std_max: float = eqx.field(static=True, default=2)
    n_samples_correct: int = eqx.field(static=True, default=100)
    eps: float = eqx.field(static=True, default=1e-6)

    def __call__(
        self,
        observation: Float[Array, "obs_dim"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "act_dim"]:
        """Stochastic action."""
        mean, log_std = self._mean_and_log_std(observation)
        std = jnp.exp(log_std)
        noise = jr.normal(key, shape=mean.shape)
        return jnp.tanh(mean + std * noise)

    def sample_and_log_prob(
        self,
        observation: Float[Array, "obs_dim"],
        *,
        key: PRNGKeyArray,
    ) -> tuple[
        Float[Array, "act_dim"],
        Float[Array, ""],
    ]:
        dist = self._distribution(observation)
        action, log_prob = dist.sample_and_log_prob(seed=key)
        return action, log_prob.sum()

    def mean_action(
        self,
        observation: Float[Array, "obs_dim"],
    ) -> Float[Array, "act_dim"]:
        """Standard deterministic action: tanh(mean)."""
        mean, _ = self._mean_and_log_std(observation)
        return jnp.tanh(mean)

    def corrected_mean_action(
        self,
        observation: Float[Array, "obs_dim"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "act_dim"]:
        """Monte Carlo estimate of E[tanh(N(mean, std))]."""
        dist = self._distribution(observation)
        samples = dist.sample(
            seed=key,
            sample_shape=(self.n_samples_correct,),
        )
        return samples.mean(axis=0)

    def _mean_and_log_std(
        self,
        observation: Float[Array, "obs_dim"],
    ) -> tuple[
        Float[Array, "act_dim"],
        Float[Array, "act_dim"],
    ]:
        output = self.net(observation)  # type: ignore[operator]
        mean, log_std = jnp.split(output, 2, axis=-1)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def _distribution(
        self,
        observation: Float[Array, "obs_dim"],
    ) -> distrax.Transformed:
        mean, log_std = self._mean_and_log_std(observation)
        std = jnp.exp(log_std)
        base = distrax.Normal(mean, std)
        return distrax.Transformed(base, distrax.Tanh())


# class TanhGaussianPolicyEnsemble(TanhGaussianPolicy):
    """Ensemble of Tanh-Gaussian policies: tanh(N(mu_i, sigma_i)) for each member i.

    Inherits _mean_and_log_std, mean_action, and _distribution from TanhGaussianPolicy —
    they work unchanged because Ensemble.__call__ returns [n_ensemble, 2*act_dim], so the
    split and clip produce [n_ensemble, act_dim] tensors automatically.

    Only __call__ and sample_and_log_prob are overridden to give each member its own key.
    """

    net: Ensemble

    def __call__(
        self,
        observation: Float[Array, "obs_dim"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "n_ensemble act_dim"]:
        mean, log_std = self._mean_and_log_std(
            observation
        )  # [n_ensemble, act_dim] each
        std = jnp.exp(log_std)
        keys = jr.split(key, self.net.n_ensemble)
        noise = jax.vmap(lambda k: jr.normal(k, shape=mean.shape[1:]))(keys)
        actions = jnp.tanh(mean + std * noise)
        return actions

    def sample_and_log_prob(
        self,
        observation: Float[Array, "obs_dim"],
        *,
        key: PRNGKeyArray,
    ) -> tuple[
        Float[Array, "n_ensemble act_dim"],
        Float[Array, "n_ensemble"],
    ]:
        mean, log_std = self._mean_and_log_std(
            observation
        )  # [n_ensemble, act_dim] each
        std = jnp.exp(log_std)
        keys = jr.split(key, self.net.n_ensemble)

        def _sample(mean_i, std_i, key_i):
            dist = distrax.Transformed(distrax.Normal(mean_i, std_i), distrax.Tanh())
            action, log_prob = dist.sample_and_log_prob(seed=key_i)
            return action, log_prob.sum()

        actions, log_probs = jax.vmap(_sample)(mean, std, keys)
        return actions, log_probs

def get_optimizer(
    net: eqx.Module | Array,
    lr: float,
    version: Literal["adam", "adamw"] = "adam",
    max_grad_norm: float | None = None,
) -> tuple[optax.GradientTransformation, optax.OptState]:
    if version == "adam":
        base_optim = optax.adam(lr)
    elif version == "adamw":
        base_optim = optax.adamw(lr)
    else:
        raise ValueError(f"{version} is not supported")

    if max_grad_norm is not None:
        optim = optax.chain(optax.clip_by_global_norm(max_grad_norm), base_optim)
    else:
        optim = base_optim

    opt_state = optim.init(eqx.filter(net, eqx.is_array))
    return optim, opt_state

def _init_target(model: eqx.Module) -> eqx.Module:
    arrays, static = eqx.partition(model, eqx.is_array)
    # JAX arrays are immutable values, so there is no mutable state shared between
    # `model` and the returned target. eqx.combine constructs a fresh pytree (new
    # Python object identity) while the underlying array buffers are shared — this
    # is safe and zero-copy. The identity map is intentional.
    return eqx.combine(jax.tree.map(lambda x: x, arrays), static)

def _init_actor(config: BaseModelConfig, d_obs: int, d_act: int, key: PRNGKeyArray):
    cfg = config.network.actor
    actor = Policy(
        net = eqx.nn.MLP(
            in_size=d_obs,
            out_size=d_act,
            width_size=cfg.hidden_dim,
            depth=cfg.depth,
            final_activation=jnp.tanh,
            key=key,
        )
    )
    # if cfg.n_ensemble == 1:
    #     actor = TanhGaussianPolicy(
    #         eqx.nn.MLP(
    #             in_size=d_obs,
    #             out_size=2 * d_act,
    #             width_size=cfg.hidden_dim,
    #             depth=cfg.depth,
    #             key=key,
    #         )
    #     )
    # else:
    #     actor = TanhGaussianPolicyEnsemble(
    #         Ensemble(
    #             lambda k: eqx.nn.MLP(
    #                 in_size=d_obs,
    #                 out_size=2 * d_act,
    #                 width_size=cfg.hidden_dim,
    #                 depth=cfg.depth,
    #                 key=k,
    #             ),
    #             n_ensemble=cfg.n_ensemble,
    #             key=key,
    #         )
    #     )
    return actor, (_init_target(actor) if cfg.has_target else None)


def _init_critic(config: BaseModelConfig, d_obs: int, d_act: int, key: PRNGKeyArray):
    cfg = config.network.critic
    critic = QFunctionEnsemble(
        Ensemble(
            lambda k: eqx.nn.MLP(
                in_size=d_obs + d_act,
                out_size=1,
                width_size=cfg.hidden_dim,
                depth=cfg.depth,
                key=k,
            ),
            n_ensemble=cfg.n_ensemble,
            key=key,
        )
    )
    return critic, (_init_target(critic) if cfg.has_target else None)

REGISTRY: dict[str, tuple[type, type]] = {}


def register(name: str, config_cls: type):
    def decorator(model_cls: type) -> type:
        REGISTRY[name] = (model_cls, config_cls)
        return model_cls

    return decorator

def compute_and_apply_gradients(
    model: eqx.Module,
    loss_fn: Callable[[eqx.Module], tuple[Scalar, dict]],
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[eqx.Module, optax.OptState, Scalar, dict]:
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, new_opt_state = optimizer.update(
        grads, opt_state, params=eqx.filter(model, eqx.is_array)
    )
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss, aux

class ActorCritic(Agent):
    """Generic base class for actor-critic agents.

    Subclasses implement act, actor_loss, critic_loss, and learn.
    """

    actor: Policy
    critic: ValueFunction

    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState

    @eqx.filter_jit
    def act(
        self, observation: Float[Array, "obs_dim"], *, key: PRNGKeyArray
    ) -> Float[Array, "act_dim"]:
        return self.actor(observation, key=key)  # type: ignore[operator]

    @abstractmethod
    def actor_loss(
        self, actor: Policy, batch: dict[str, Array], key: PRNGKeyArray
    ) -> tuple[Scalar, dict]:
        """Compute actor loss. Gradient is taken w.r.t. actor."""
        raise NotImplementedError

    @abstractmethod
    def critic_loss(
        self, critic: ValueFunction, batch: dict[str, Array], key: PRNGKeyArray
    ) -> tuple[Scalar, dict]:
        """Compute critic loss. Gradient is taken w.r.t. critic."""
        raise NotImplementedError

    @classmethod
    def _base_fields(
        cls, config: BaseModelConfig, *, d_obs: int, d_act: int, key: PRNGKeyArray
    ) -> dict:
        """Build the shared actor/critic fields dict. Subclasses call this to extend."""
        actor_key, critic_key = jr.split(key)
        actor, target_actor = _init_actor(config, d_obs, d_act, actor_key)
        critic, target_critic = _init_critic(config, d_obs, d_act, critic_key)
        actor_optimizer, actor_opt_state = get_optimizer(
            actor,
            config.optimizer.actor.lr,
            version=config.optimizer.actor.optim,
            max_grad_norm=config.optimizer.actor.max_grad_norm,
        )
        critic_optimizer, critic_opt_state = get_optimizer(
            critic,
            config.optimizer.critic.lr,
            version=config.optimizer.critic.optim,
            max_grad_norm=config.optimizer.critic.max_grad_norm,
        )
        return {
            "actor": actor,
            "target_actor": target_actor,
            "critic": critic,
            "target_critic": target_critic,
            "actor_optimizer": actor_optimizer,
            "critic_optimizer": critic_optimizer,
            "actor_opt_state": actor_opt_state,
            "critic_opt_state": critic_opt_state,
        }

    @classmethod
    def _create_kwargs(
        cls, config: BaseModelConfig, *, d_obs: int, d_act: int, key: PRNGKeyArray
    ) -> dict:
        f = cls._base_fields(config, d_obs=d_obs, d_act=d_act, key=key)
        return {
            "actor": f["actor"],
            "critic": f["critic"],
            "actor_optimizer": f["actor_optimizer"],
            "critic_optimizer": f["critic_optimizer"],
            "actor_opt_state": f["actor_opt_state"],
            "critic_opt_state": f["critic_opt_state"],
        }

    @classmethod
    def create(
        cls, config: BaseModelConfig, *, d_obs: int, d_act: int, key: PRNGKeyArray
    ) -> "ActorCritic":
        return cls(**cls._create_kwargs(config, d_obs=d_obs, d_act=d_act, key=key))
