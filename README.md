# Enhancing-Classical-Motion-Planners-Using-RL-with-Safety-Guarantees

## Main Algorithm (TD3+OE)

Modification of TD3+BC: https://github.com/sfujim/TD3_BC , to include Online Expert (instead of behavioral cloning) and explicit trust region.

1) Actor loss (TD3+OE)

We replace TD3+BC’s BC term with an online-expert penalty and cap the adaptive weight to enforce a trust region:

λ~=min⁡{λ,λ0},Lactor=Es∼D[ λ~ Qθ ⁣(s,πϕ(s))−12∥πϕ(s)−πexpert(s)∥22].
λ
~
=min{λ,λ
0
	​

},L
actor
	​

=E
s∼D
	​

[
λ
~
Q
θ
	​

(s,π
ϕ
	​

(s))−
2
1
	​

∥π
ϕ
	​

(s)−π
expert
	​

(s)∥
2
2
	​

].

Weights.

λ=α⟨∣Qθ∣⟩B=α1N∑(si,ai)∈B∣Qθ(si,ai)∣,λ0=2 εmax⁡gmax⁡.
λ=
⟨∣Q
θ
	​

∣⟩
B
	​

α
	​

=
N
1
	​

∑
(s
i
	​

,a
i
	​

)∈B
	​

∣Q
θ
	​

(s
i
	​

,a
i
	​

)∣
α
	​

,λ
0
	​

=
g
max
	​

2ε
max
	​

	​

.

α
α is a small constant (e.g., 
2.5
2.5) for scale normalisation.

εmax⁡
ε
max
	​

 sets the desired safety radius (max allowed deviation from the expert).

gmax⁡
g
max
	​

 is the action-gradient cap enforced via the critic (below).
This matches the formulation and trust-region cap introduced in the paper. 
<img width="300" height="600" alt="algorithm_td3_oe" src="https://github.com/user-attachments/assets/427271dd-d29f-43f9-82f1-38a7ece8f50c" />
