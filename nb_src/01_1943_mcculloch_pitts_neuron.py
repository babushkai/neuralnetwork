#%% md
# 1943 — The McCulloch–Pitts Neuron

**Paper:** Warren McCulloch & Walter Pitts, *"A Logical Calculus of the Ideas Immanent in Nervous Activity"* (1943).

The first mathematical model of a neuron. It has **binary inputs**, **fixed (hand-set) weights**, and a **threshold**:

$$
y = \begin{cases} 1 & \text{if } \sum_i w_i x_i \ge \theta \text{ and no inhibitory input is active} \\ 0 & \text{otherwise} \end{cases}
$$

Key points:
- There is **no learning** — a human chooses the weights and threshold.
- A single unit can compute AND, OR and NOT.
- *Networks* of units can compute any Boolean function (including XOR), which made the brain look like a logic machine.

No GPU needed — pure NumPy.
#%% md
![Diagram: McCulloch–Pitts neuron (1943)](https://raw.githubusercontent.com/babushkai/neuralnetwork/main/assets/diagrams/01_1943_mcculloch_pitts_neuron.png)
#%%
import numpy as np
import itertools
#%% md
## A McCulloch–Pitts unit
Inputs are either *excitatory* (weight +1) or *inhibitory* (absolute veto, as in the original paper).
#%%
class MPNeuron:
    def __init__(self, threshold, n_excitatory, n_inhibitory=0):
        self.theta = threshold
        self.n_exc = n_excitatory
        self.n_inh = n_inhibitory

    def __call__(self, excitatory, inhibitory=()):
        excitatory, inhibitory = np.asarray(excitatory), np.asarray(inhibitory)
        if inhibitory.size and inhibitory.any():   # any active inhibitory input vetoes firing
            return 0
        return int(excitatory.sum() >= self.theta)

def truth_table(name, fn, n_inputs):
    print(f"{name}")
    for x in itertools.product([0, 1], repeat=n_inputs):
        print("  ", x, "->", fn(*x))
#%% md
## Logic gates from single units
#%%
AND = MPNeuron(threshold=2, n_excitatory=2)
OR  = MPNeuron(threshold=1, n_excitatory=2)
NOT = MPNeuron(threshold=0, n_excitatory=0, n_inhibitory=1)   # fires unless inhibited

truth_table("AND", lambda a, b: AND([a, b]), 2)
truth_table("OR",  lambda a, b: OR([a, b]), 2)
truth_table("NOT", lambda a: NOT([], [a]), 1)
#%% md
## XOR needs a *network*
No single threshold unit can compute XOR (we prove this in the 1969 notebook). But two layers can:

$$\text{XOR}(a,b) = \text{AND}(\text{OR}(a,b),\ \text{NOT AND}(a,b))$$

Using an inhibitory connection: fire if `OR(a,b)` is on, unless `AND(a,b)` is on.
#%%
def XOR(a, b):
    h_or = OR([a, b])
    h_and = AND([a, b])
    out = MPNeuron(threshold=1, n_excitatory=1, n_inhibitory=1)
    return out([h_or], [h_and])

truth_table("XOR (2-layer MP network)", XOR, 2)
#%% md
## Majority vote and a tiny "sequential" circuit
A 3-input majority gate is one unit with threshold 2. MP networks with loops can also hold memory — here a set/reset latch.
#%%
MAJ = MPNeuron(threshold=2, n_excitatory=3)
truth_table("MAJORITY(a,b,c)", lambda a, b, c: MAJ([a, b, c]), 3)

# A latch: the unit excites itself (recurrent connection); 'reset' is inhibitory.
latch = MPNeuron(threshold=1, n_excitatory=2, n_inhibitory=1)
state = 0
events = [("set", 1, 0), ("idle", 0, 0), ("idle", 0, 0), ("reset", 0, 1), ("idle", 0, 0)]
for name, s, r in events:
    state = latch([s, state], [r])
    print(f"{name:5s} -> memory = {state}")
#%% md
## Takeaways
- A neuron = weighted sum + threshold. Every later model in this repo still has that core.
- The missing ingredient is **learning the weights** from data — that arrives with the perceptron (1958).
