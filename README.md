# UoM COMP90054 AI Planning for Autonomy - Pacman Project

# Design Choices

We have used four ways to implement our agents, including heuristics with weighted features, Minimax with weighted features, Astar search, and Mixture of Minimax and Heuristic.

## General Comments

The general concepts of this project are:
1. Agents need to search for food in enemies' territory.
2. Agents need to avoid ghosts when in enemies' territory.
3. Agents need to bring food back to their territory after eating certain amount of food.
4. Agents need to avoid dead-ends when there are ghosts around.
5. One of the agents needs to hunt for invaders.   

## Comments per topic

### Heuristic agents
We start from the basic Heuristic agents using the heuristic function of many features and their corresponding weights to approximate our Q value and select the best action. By adjusting weights and decision tree logic, it can fulfil the general comments of 1-5 but sometimes not smart enough. 

### Astar agents

When we found out that we can only beat staff_medium team no matter how hard we've tried to hard-code our decision trees, we decided to move to Astar search approach, which can search deeply through many steps according to the given compute time and have a better planning outcome. It can fulfil the general comments of 1-5 with depth one. But when the search depth is greater than one, the agent might be stuck in a deadlock of going back and forth repeatedly. This is caused by changing decisions rapidly.

### Minimax agents

We move to build minimax because we want to both consider our best move and the move of our opponents and pick the comparatively better choice. It is good for attacking but it is relatively weak in defending. It can fulfil the general comments of 1-4.

### Mixed agents with Minimax and Heuristic

Finally, we decided to combine both Mininmax and Heuristic agent to see whether two approaches can work perfectly together. we want to combine the advantage of both agents to have synergy effect. Minimax agent is mainly responsible for attacking while the Heuristic agent is the one to hunt the invaders. It can fulfil the general comments of 1-5 better.

