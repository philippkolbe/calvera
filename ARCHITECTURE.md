# Architecture Specification of the library

## Tasks

### Datasets
- MNIST
- STATLOG
- Some kind of RAG dataset
- Synthetic dataset (wheel bandit, generation functions)

### Feedback -- offline vs. online
- Offline feedback
- store probabilities for logged feedback-based training


### Algorithms
*NOTE*: everything is contextual
#### Exploration Strategies
- Linear Bandits (LinUCB, LinTS)
- ($\epsilon$)-greedy
- NeuralUCB (UCB with gradients)
- NeuralTS
- Combinatorial Bandits (maybe we need to figure the integration of this out)

#### Architectures
- Bootstrap
- Neural Networks
- Low Rank Adaption of Neural Networks

---

# Architecture specifics
 
