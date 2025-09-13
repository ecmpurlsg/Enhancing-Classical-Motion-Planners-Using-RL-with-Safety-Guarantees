# Enhancing Classical Motion Planners with RL (Safety-Guaranteed)

A practical framework that augments ROS1 navigation with **TD3+OE** (TD3 with Online Expert and an explicit trust region) for safer, more efficient motion planning.

## What’s inside
- 🧭 **ROS1 navigation** /env/ files.
- 🤖 **Training pipeline** for TD3+OE on top of the ROS1 stack.

## Main Algorithm — TD3+OE
TD3 enhanced with an online expert term (instead of offline BC) and a capped actor weight to enforce a trust region; critics are regularized with a hinge gradient penalty for stability.


Modification of TD3+BC: https://github.com/sfujim/TD3_BC , to include Online Expert (instead of behavioral cloning) and explicit trust region.


## 1) Actor loss (TD3+OE)

We replace TD3+BC’s BC term with an **online-expert** penalty and cap the adaptive weight to enforce a trust region:

$$
\tilde{\lambda}=\min\{\lambda,\lambda_0\},\qquad
L_{\text{actor}}
=\mathbb{E}_{s\sim\mathcal{D}}\left[
\tilde{\lambda}\,Q_{\theta}\big(s,\pi_{\phi}(s)\big)
-\tfrac{1}{2}\,\big\|\pi_{\phi}(s)-\pi_{\text{expert}}(s)\big\|_2^{2}
\right].
$$

**Weights**

$$
\lambda=\frac{\alpha}{
\frac{1}{N}\displaystyle\sum_{(s_i,a_i)\in\mathcal{B}}
\big|Q_{\theta}(s_i,a_i)\big|},
\qquad
\lambda_{0}=\frac{2\varepsilon_{\max}}{g_{\max}}.
$$

* $\varepsilon_{\max}$: desired trust-region radius (max allowed deviation from the expert).
* $g_{\max}$: cap on the critic’s action-gradient norm (enforced below).

> In practice we compute $\lambda$ per minibatch (or with an EMA of $|Q|$) and then clamp by $\lambda_0$ to get $\tilde{\lambda}$.

---

## 2) Critic loss with hinge gradient-penalty

For a minibatch the critic objective is

$$
\begin{aligned}
L_Q
&=\frac{1}{N}\sum_{i=1}^{N}\Big[
\big(y_i-Q_{\theta_1}(s_i,a_i)\big)^2
+\big(y_i-Q_{\theta_2}(s_i,a_i)\big)^2\Big] \\
&\quad+\;\beta_{\mathrm{gp}}
\Big[\tfrac12\!\big(\|\nabla_a Q_{\theta_1}(s_i,a)\|_2
+\|\nabla_a Q_{\theta_2}(s_i,a)\|_2\big)-g_{\max}\Big]_+^{2},
\end{aligned}
$$

where $[x]_+=\max(0,x)$ and the action-gradients are evaluated at $a=a_i$.

* This **averaged hinge penalty** keeps both critics’ $\|\nabla_a Q\|\le g_{\max}$ (a Lipschitz cap), which makes the actor’s trust-region bound meaningful and stabilizes targets computed with $\min(Q_1',Q_2')$.

---

## Theorem (trust-region / deviation guarantee)

Assume the critic is $L$-Lipschitz in the action with respect to the $\ell_2$ norm and $L\le g_{\max}$ (enforced by the hinge penalty).
Let $\pi_\phi^\star$ be any stationary point of $L_{\text{actor}}$ with $\tilde{\lambda}=\min\{\lambda,\lambda_0\}$. Then, for every state $s$,

$$
\big\|\pi_\phi^\star(s)-\pi_{\text{expert}}(s)\big\|
\le\frac{\tilde{\lambda}L}{2}
\le\frac{\lambda_0L}{2}
=\varepsilon_{\max}.
$$

---


<img width="300" height="600" alt="algorithm_td3_oe" src="https://github.com/user-attachments/assets/427271dd-d29f-43f9-82f1-38a7ece8f50c" />

## Simulation Top view


https://github.com/user-attachments/assets/8c88e1cc-3d3a-4da2-8dc1-d89406a8aeb6




