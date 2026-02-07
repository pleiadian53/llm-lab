---
title: "TD(λ): From One-Step Bootstrapping to Eligibility Traces"
author: "LLM Lab"
date: "February 2026"
---

# TD(λ): From One-Step Bootstrapping to Eligibility Traces

**How temporal difference learning generalizes across bootstrapping horizons — and why it matters for modern RL**

> *Prerequisites: Familiarity with SARSA and Q-learning update rules. See the companion tutorial [On-Policy vs Off-Policy RL](summary.md) for background.*

---

## 1. Motivation: What Are SARSA and Q-learning Really Doing?

The standard SARSA and Q-learning updates look like this:

- **SARSA:** $Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s,a) \right]$
- **Q-learning:** $Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s,a) \right]$

Both updates observe **exactly one reward** $r_t$ and then immediately **bootstrap** — they substitute a value estimate $Q(\cdot)$ for all future rewards rather than waiting to see what actually happens.

In Sutton & Barto's terminology, these are **TD(0)** methods: the "0" indicates that the agent bootstraps after zero additional steps of real experience beyond the immediate reward.

A natural question arises: *what if we waited longer before bootstrapping?* What if we used two, three, or even all remaining rewards before plugging in a value estimate? This is exactly what **n-step TD** and **TD(λ)** address.

### A common point of confusion

The "0" in TD(0) refers to the **number of future steps observed before bootstrapping**, not the number of time steps in the update equation. TD(0) uses one reward and bootstraps immediately. TD(1) would use the **full Monte Carlo return** — all rewards until the episode ends — with no bootstrapping at all. The index controls the **horizon before bootstrapping**, not the step count.

| Label | Rewards Used | Bootstrapping |
|-------|-------------|---------------|
| TD(0) | One ($r_t$) | Immediate |
| TD($n$) | $n$ rewards ($r_t, \ldots, r_{t+n-1}$) | After $n$ steps |
| TD(1) / Monte Carlo | All remaining | None |

---

## 2. The Core Object: The Return

In an episodic MDP, the **true return** from time step $t$ is:

$$G_t \equiv \sum_{k=0}^{T-t-1} \gamma^k \, r_{t+k}$$

where $T$ is the terminal time step and $\gamma \in [0,1]$ is the discount factor.

This is what *every* value-based RL method is trying to estimate. The differences between methods come down to **how they approximate $G_t$** when the full sequence of future rewards is not yet available.

---

## 3. N-Step Returns: Choosing a Bootstrapping Horizon

### 3.1 Definition

The **n-step return** uses $n$ real rewards and then bootstraps from a value estimate:

$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k \, r_{t+k} + \gamma^n \, V(s_{t+n})$$

If the episode terminates before step $t+n$, the bootstrap term vanishes and the return reduces to the actual Monte Carlo return over the remaining steps.

### 3.2 Special cases

- **$n = 1$:** This is TD(0). One reward, immediate bootstrap: $G_t^{(1)} = r_t + \gamma V(s_{t+1})$.
- **$n = T - t$:** This is Monte Carlo. All rewards, no bootstrap: $G_t^{(T-t)} = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$.

Any fixed $n$ between these extremes gives an **n-step TD method** (sometimes written TD($n$) or TD-$k$).

### 3.3 N-step SARSA and n-step Q-learning

The n-step idea applies directly to action-value methods:

- **N-step SARSA** uses the target: $G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n Q(s_{t+n}, a_{t+n})$
- **N-step Q-learning** uses: $G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q(s_{t+n}, a')$

These are well-defined and used in practice (e.g., 3-step or 5-step returns in Rainbow DQN). The update rule is the same as standard SARSA or Q-learning, but with $G_t^{(n)}$ replacing the one-step target.

### 3.4 The bias–variance trade-off

Each n-step return sits at a different point on the bias–variance spectrum:

- **Small $n$ (e.g., TD(0)):** Low variance (less randomness from future rewards), but high bias (the bootstrap value $V(s_{t+n})$ may be inaccurate).
- **Large $n$ (e.g., Monte Carlo):** Low bias (using real rewards), but high variance (the sum of many stochastic rewards is noisy).

This trade-off motivates the question: *is there a principled way to combine all n-step returns rather than picking a single $n$?*

---

## 4. TD(λ): A Principled Mixture of All Horizons

### 4.1 The λ-return

TD(λ) does not commit to a single bootstrapping horizon. Instead, it forms a **geometrically weighted mixture** of all n-step returns:

$$G_t^{(\lambda)} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \, G_t^{(n)}$$

where $\lambda \in [0, 1]$ controls the weighting:

- **Small $\lambda$:** Short-horizon returns (small $n$) receive most of the weight. The estimate is biased but low-variance.
- **Large $\lambda$:** Long-horizon returns dominate. The estimate approaches Monte Carlo — low bias, high variance.

### 4.2 Limiting cases

- **$\lambda = 0$:** Only the 1-step return survives: $G_t^{(0)} = G_t^{(1)} = r_t + \gamma V(s_{t+1})$. This is TD(0).
- **$\lambda \to 1$:** The weights spread evenly across all horizons, and the λ-return converges to the full Monte Carlo return $G_t$. This is TD(1).

### 4.3 Why this works

The geometric weighting $(1 - \lambda) \lambda^{n-1}$ is not arbitrary. It ensures the weights sum to 1 (a proper convex combination) and provides a smooth, single-parameter interpolation between the two extremes of the bias–variance spectrum. TD(λ) is not "fancy TD" — it is **variance-controlled estimation of the return**.

---

## 5. From Theory to Computation: Eligibility Traces

### 5.1 The computational problem

The λ-return definition above is *conceptual*: it requires knowing all future rewards before computing the update, which defeats the purpose of online learning. Eligibility traces solve this by providing an **incremental, online implementation** that is mathematically equivalent.

### 5.2 State-value TD(λ)

Define the one-step **TD error**:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

Define the **eligibility trace** for each state:

$$e_t(s) = \gamma \lambda \, e_{t-1}(s) + \mathbb{1}[s = s_t]$$

The trace $e_t(s)$ records how recently and frequently state $s$ was visited. It decays by $\gamma \lambda$ at each step and gets a bump of $+1$ whenever the agent visits $s$.

The update rule applies to **all states simultaneously**:

$$V(s) \leftarrow V(s) + \alpha \, \delta_t \, e_t(s)$$

This single update, applied at every time step, is mathematically equivalent to the full λ-return update — but computed incrementally without looking into the future.

### 5.3 Intuition

When a TD error $\delta_t$ occurs (the agent is surprised by a reward), the eligibility trace determines **which past states share the credit**. Recently visited states have high traces and receive large updates; states visited long ago have decayed traces and receive small updates. The parameter $\lambda$ controls how far back credit is assigned.

---

## 6. Action-Value TD(λ): SARSA(λ)

The same machinery extends to action-value functions by replacing $V(s)$ with $Q(s, a)$.

### 6.1 SARSA(λ) update

TD error:

$$\delta_t = r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)$$

Eligibility trace:

$$e_t(s, a) = \gamma \lambda \, e_{t-1}(s, a) + \mathbb{1}[(s, a) = (s_t, a_t)]$$

Update (applied to all state-action pairs):

$$Q(s, a) \leftarrow Q(s, a) + \alpha \, \delta_t \, e_t(s, a)$$

### 6.2 Where standard SARSA fits

Standard SARSA is simply **SARSA($\lambda = 0$)**:

- The trace decays instantly ($\gamma \cdot 0 = 0$), so only the current $(s_t, a_t)$ has a nonzero trace
- Only the current state-action pair is updated
- This recovers the familiar one-step TD(0) update

### 6.3 N-step SARSA without traces

For a fixed horizon $k$, n-step SARSA can also be implemented directly (without eligibility traces) using the target:

$$G_t^{(k)} = \sum_{i=0}^{k-1} \gamma^i \, r_{t+i} + \gamma^k \, Q(s_{t+k}, a_{t+k})$$

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( G_t^{(k)} - Q(s_t, a_t) \right)$$

This is common in practice with small fixed $k$ (e.g., 3-step SARSA). Eligibility traces can be viewed as the **efficient implementation of the infinite mixture** over all such $k$.

---

## 7. Q-Learning and TD(λ): A Subtle Complication

### 7.1 Standard Q-learning as TD(0)

The Q-learning TD error is:

$$\delta_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)$$

This is off-policy (the target uses the greedy $\max$, not the action actually taken) and TD(0) (immediate bootstrap).

### 7.2 Why Q(λ) is problematic

Naively adding eligibility traces to Q-learning **can break convergence**. The fundamental issue is a mismatch between the trace and the target:

- **Traces** propagate credit along the trajectory generated by the *behavior* policy (the policy actually followed, including exploration)
- **Targets** assume the agent will act according to the *greedy* policy (the $\max$ operator)

When the behavior policy takes a non-greedy (exploratory) action, the trace is propagating credit along a path the target policy would never have taken. This inconsistency can cause value estimates to diverge, especially with function approximation.

### 7.3 Solutions

Two approaches address this mismatch:

1. **Watkins' Q(λ)** — Cut the trace whenever a non-greedy action is taken. This is safe and guarantees convergence, but conservative: traces are frequently reset, limiting the benefit of multi-step credit assignment.

2. **Expected SARSA(λ)** — Replace the $\max$ with an expectation under the target policy. This produces smoother targets and more stable learning, effectively bridging on-policy and off-policy updates.

Because of these complications, **SARSA(λ) is far more commonly used** than Q(λ) in practice. The on-policy setting avoids the trace–target mismatch entirely.

---

## 8. The Algorithm Design Space

The on/off-policy distinction and the TD horizon are **orthogonal design axes**. Any algorithm can be placed on a grid:

| Algorithm | Policy Type | TD Horizon | Notes |
|-----------|------------|------------|-------|
| SARSA | On-policy | TD(0) | Standard one-step |
| N-step SARSA | On-policy | TD($k$) | Fixed horizon |
| SARSA(λ) | On-policy | TD(λ) | Full eligibility traces |
| Q-learning | Off-policy | TD(0) | Standard one-step |
| N-step Q-learning | Off-policy | TD($k$) | Unstable without corrections |
| Watkins' Q(λ) | Off-policy | TD(λ) | Traces cut at non-greedy actions |

Not all regions of this grid are equally stable. The combination of **off-policy learning + multi-step bootstrapping + function approximation** is the infamous **deadly triad** (Sutton & Barto, Ch. 11), which is why the literature clusters around the more stable configurations.

---

## 9. Why Multi-Step Methods Are Less Common in Practice

Given the theoretical appeal of n-step and TD(λ) methods, it is worth understanding why they appear less frequently in textbooks and implementations than their TD(0) counterparts.

**Stability.** Multi-step methods amplify the instability of the deadly triad. N-step Q-learning, in particular, sits squarely in the danger zone.

**Implementation complexity.** Eligibility traces require maintaining an additional vector of the same size as the parameter space, careful handling with experience replay buffers, and special treatment at episode boundaries.

**Historical momentum.** When Deep Q-Networks (DQN) achieved breakthrough results, the recipe was one-step TD + experience replay + target networks. Multi-step returns were reintroduced later (e.g., 3-step returns in Rainbow) but with careful engineering guardrails and off-policy corrections.

**Policy gradient methods changed the framing.** Modern actor–critic methods use Generalized Advantage Estimation (GAE), which is mathematically equivalent to TD(λ) applied to the advantage function — but phrased in policy-gradient language. TD(λ) didn't disappear; it **changed clothes**.

---

## 10. Connection to Modern RL: GAE as TD(λ) in Disguise

Generalized Advantage Estimation (Schulman et al., 2016), used in PPO and other actor–critic methods, defines the advantage estimate as:

$$\hat{A}_t^{\text{GAE}(\lambda)} = \sum_{k=0}^{\infty} (\gamma \lambda)^k \, \delta_{t+k}$$

where $\delta_{t+k} = r_{t+k} + \gamma V(s_{t+k+1}) - V(s_{t+k})$ is the one-step TD error.

This is **exactly TD(λ)** applied to the advantage function rather than the value function. The parameter $\lambda$ plays the same role: it controls the bias–variance trade-off between short-horizon (low-variance, high-bias) and long-horizon (high-variance, low-bias) advantage estimates.

So when PPO uses GAE with $\lambda = 0.95$:

- It is performing TD(λ) estimation
- The λ parameter trades off bias and variance in the advantage signal
- The conceptual machinery is identical to SARSA(λ), just applied in a policy-gradient context

---

## 11. Summary

Two orthogonal questions define the space of TD-based RL algorithms:

> **On-policy vs. off-policy:** Whose policy does the bootstrap target assume?
>
> **TD horizon (0, $k$, or λ):** How many real rewards are observed before bootstrapping?

The key takeaways:

- Standard SARSA and Q-learning are **TD(0)** — they bootstrap immediately after one reward
- **N-step** and **TD(λ)** variants exist for both algorithms and are well-defined
- TD(λ) provides a principled, single-parameter interpolation across the full bias–variance spectrum
- **Eligibility traces** make TD(λ) computationally efficient by enabling online, incremental updates
- Off-policy TD(λ) (Q(λ)) is problematic due to the trace–target mismatch; on-policy SARSA(λ) is more stable
- In modern deep RL, TD(λ) lives on as **GAE** inside actor–critic methods like PPO

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Chapters 7 and 12.
2. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
3. Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards.* (Watkins' Q(λ))
4. Hessel, M., et al. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning.* AAAI 2018. (Multi-step returns in DQN)

---

*See also: [On-Policy vs Off-Policy RL](summary.md) for the SARSA vs Q-learning comparison that this tutorial builds upon.*
