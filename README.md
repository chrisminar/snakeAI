# Snake reinforcement learning.
Trains a neural net to play the arcade game snake using reinforcement learning.  
100 points/food, -1 point/move.
## Revamped method
Neural network and training was overhauled as well as playing many more games per generation made possible via numpy acceleration of snake game playing. A percentage of moves are also taken randomly while training.

### Decreasing exploration rate
![](media/gifs/4x4_fast_explore.gif)  
Examples of neural net performance at each generation. Notice that at earlier generations the snake can get stuck in loops that while safe, will result in a timeout.  
![](media/pictures/4x4_decreasing_exploration_rate.png)  
Performance as a function of generation with a exploration rate of 20% that decreases to 5% over time. Notice that the training does not converge at generation 11!  
### Constant exploration rate
![](media/gifs//4x4_nn_optimization_1.gif)  
Examples of neural net performance at each generation.  
![](media/pictures/4x4_constant_exploration_rate.png)  
Performance as a function of generation with a constant exploration rate of 10% while the score is below 1000. Notice the training does not converge for several generations between 10 and 20.
