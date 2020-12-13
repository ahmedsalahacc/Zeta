# Monte Carlo Tree Search for Reinforcement Learning
## Selection
* Use UCB1 (Upper Confidence Bound) Algorithm to select the node with the highest estimated value.
* If the node is Unvisited, choose a random node to rollout (simulate) and calculate the value at the end of the simulation.
## Expansion
* In the expansion phase, and after selecting the optimal state (using the selection step), we will expand (add new states initialized with zeros) the selected node if it was visited before.
* The Expansion is based on the value network of the pretrained model
## Simulation (Rolling out)
* In this part, moves are performed by choosing nodes or strategies until a result or predefined state is achieved (termination).
## Backpropagation
* Update each state recursively reaching the root state with the new score and visit count