# reinforcement_learning
Reinforcement learning experiments and demos

This repository contains some command-line demos that I wrote in order to understand how reinforcement learning works, and to try out various experiments and alternate implementations of existing methodologies. The following is an explanation of what you'll find here.

The qtables/ directory contains a simple qtables demo based on a simple game. The game presents a 5x5 grid. Four positions on the grid are occupied by "soldiers". Moving onto one of those spaces ends the game with negative reward. One space contains a goal ("princess"). Moving onto that space ends with game with a positive reward. The player always starts at the top left (0,0) position. The demo performs an epsilon greedy strategy for learning to play. Sleep calls are included to start the demo slowly, and increase speed over time. The display itself shows the values in the qtable, the bellman equation being calculated, and some information about the game itself.

keras_parachute_dqn and pytorch_dqn_parachute are vanilla DQN implementations using the respective frameworks. These demos illustrate a simple game where a paddle at the bottom of the screen must move left or right in order to catch falling objects. The idea comes from the old Nintendo Game and Watch "Parachute" game. The demo shows a human readable representation of the game, as it is being played, along with statistics about what is happening during training. As with all demos here, everything happens in the console.

The "DRQN" implementations utilize a recurrent model instead of a simple fully-connected arthictecture. I have included a number of tricks that attempt to stabilize and expedite training. These include saving replay sequences that end in reward states, only using exploration when reaching the average number of steps played over the last 50 games, and rebooting exploration (setting epsilon to a non-zero value) if the training starts to degrade (based on the average score attained over the last 50 episodes). It sort of works for the parachute game, but not really for the others.

Included are a few other games, which are all work in progress.

- worm is a game similar to the "snake" game on Nokia phones
- tetris is tetris
- multiply is a really simple experiment that attempts to learn the times tables by guessing answers

keras_HER is a keras implementation of Hindsight Experience Replay (since I couldn't find one when I looked)
pytorch_test_GRU is a scratch jupyter notebook I have been using to play with pytorch and creating random neural net architectures.

pytorch_question_answer implements a game that presents the player with a series of statements and a question, that can be answered by reading the statements. I am fully aware that there are neural reasoning architectures for solving these types of games, but out of curiosity, I've been looking at whether a more generic reinforcement learning approach might just magically work for this type of problem. This approach only seems to get about 20% of the answers right.

