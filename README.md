# N-step-Dueling-DDQN-PER-Pacman
> Using N-step dueling DDQN with PER for learning how to play a Pacman game

## Summary

DeepMind published its famous paper Playing Atari with Deep Reinforcement Learning, in which a new algorithm called DQN was implemented. It showed that an AI agent could learn to play games by simply watching the screen without any prior knowledge about the game. Also, I have added a few enhancement to the vanilla DQN from various papers and tested it on the MsPacman-v4 OpenAI's environment.


<img src="./Figure_1.png" align="middle">

## Demo

<img src="./overview.gif" width="256" align="middle">

<hr />

## Preprocessing

  - For keeping the downsampled image from distortion, I have dilated the pixels with a (3,3) kernel two times.
  - Downsample the game image to the size 88x80.
  - Change the color of Pacman's pixels for precise observation.

## DQN

  - Instead of stacking four images and feeding them to the network, I'm taking an average of 4 recent images.
  - The main network is being updated each 1000 steps.
  - The replay buffer size is 100000.
  - The network architecture is:
  ```
  __________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 88, 80, 1)    0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 22, 20, 32)   2080        input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 11, 10, 64)   32832       conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 11, 10, 64)   36928       conv2d_2[0][0]                   
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 7040)         0           conv2d_3[0][0]                   
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 5)            35205       flatten_1[0][0]                  
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 1)            0           dense_2[0][0]                    
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            7041        flatten_1[0][0]                  
__________________________________________________________________________________________________
subtract_1 (Subtract)           (None, 5)            0           dense_2[0][0]                    
                                                                 lambda_1[0][0]                   
__________________________________________________________________________________________________
add_1 (Add)                     (None, 5)            0           dense_1[0][0]                    
                                                                 subtract_1[0][0]                 
==================================================================================================
Total params: 114,086
Trainable params: 114,086
Non-trainable params: 0
__________________________________________________________________________________________________
---------------------------------------------------------------------------
```

## Double DQN

For updating the Q values in the max operator, DQN uses the same values both to select and to evaluate an action. This makes it more likely to select overestimated values, resulting in overoptimistic value estimates. In order to solve this issue, we can use the target network as a value estimator and the main network as an action selector.

## N-Step Return

DQN accumulates a single reward and then uses the greedy action at the next step to bootstrap. Alternatively, forward-view multi-step targets can be used and bootstrap after few steps (5 steps here).

## Dueling Network

The dueling architecture can learn which states are valuable for each state without learning the effect of each action. This is particularly useful in states where its actions in no relevant way affect the environment. Also, for a more stable optimization, we use an average baseline for Q evaluation.

## Prioritized Experience Replay

Lastly, we can prioritize the episode by the magnitude of a transition’s TD error. Moreover, to overcome the issue of replaying a subset of transitions more frequently, we will use a stochastic sampling method that interpolates between pure greedy prioritization and uniform random sampling. I used a min-heap and chose about 60% by the TD error priority and 40% uniformly.

## Related Papers

01. [V. Mnih et al., "Human-level control through deep reinforcement learning." Nature, 518
(7540):529–533, 2015.](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
02. [van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning." arXiv preprint arXiv:1509.06461, 2015.](https://arxiv.org/pdf/1509.06461.pdf)
03. [T. Schaul et al., "Prioritized Experience Replay." arXiv preprint arXiv:1511.05952, 2015.](https://arxiv.org/pdf/1511.05952.pdf)
04. [Z. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning." arXiv preprint arXiv:1511.06581, 2015.](https://arxiv.org/pdf/1511.06581.pdf)
05. [R. S. Sutton, "Learning to predict by the methods of temporal differences." Machine learning, 3(1):9–44, 1988.](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)

<hr />

## Author

  - Soheil Changizi ( [@cocolico14](https://github.com/cocolico14) )


## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details


