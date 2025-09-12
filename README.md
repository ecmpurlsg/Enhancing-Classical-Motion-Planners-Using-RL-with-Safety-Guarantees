# Enhancing-Classical-Motion-Planners-Using-RL-with-Safety-Guarantees

## Main Algorithm (TD3+OE)

Modification of TD3+BC: https://github.com/sfujim/TD3_BC , to include Online Expert (instead of behavioral cloning) and explicit trust region.

\paragraph{1) Actor loss (TD3+OE).}
We replace TD3+BC’s BC term with an \emph{online-expert} penalty and cap the adaptive weight to enforce a trust region:
\[
\tilde{\lambda}=\min\{\lambda,\lambda_0\},\qquad
L_{\mathrm{actor}}
=\mathbb{E}_{s\sim\mathcal{D}}\!\left[
\tilde{\lambda}\,Q_{\theta}\!\big(s,\pi_{\phi}(s)\big)
-\tfrac{1}{2}\,\big\|\pi_{\phi}(s)-\pi_{\mathrm{expert}}(s)\big\|_2^{2}
\right].
\]

\noindent\textbf{Weights.}
\[
\lambda
=\frac{\alpha}{\frac{1}{N}\displaystyle\sum_{(s_i,a_i)\in\mathcal{B}}
\big|Q_{\theta}(s_i,a_i)\big|},
\qquad
\lambda_{0}=\frac{2\,\varepsilon_{\max}}{g_{\max}}.
\]

<img width="300" height="600" alt="algorithm_td3_oe" src="https://github.com/user-attachments/assets/427271dd-d29f-43f9-82f1-38a7ece8f50c" />
