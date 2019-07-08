# Overview

The purpose of this project is to implement Pac-Man Autonomous Agent that can play and compete in a tournament, the detail description of the tournament can be found from http://ai.berkeley.edu/contest.html.

![General_Pic](/uploads/d024ed699934497368562ea8d14d283a/General_Pic.PNG)


This wiki is mainly divided into three parts:
1. Challenges and Decision Making

2. Techniques used:
 * Scan Map(BFS)
 * A* Agents
 * Q-approximate Agents

3. Experimental
4. Possible Improvements


## Youtube presentation
Please click [HERE](https://www.youtube.com/watch?v=0JaM9lpQVBs&feature=youtu.be) to watch the video

##



# Technique used
## A* agents
#### Basic idea
A star search is used in both attacker and defender. It is the major techniques we used to compete in the tournament and usually remain in the top 10 and sometimes up to the top3.

The only thing we care about ín the A* algorithm is to define the goal state of different problem and heuristic. In the different game state, we would have a different goal, and the heuristic is different for defender and attacker.

It is worth mentioning that in the implementation, we only execute the first action that calculated by A*, and in next gameState, we do the calculation once again.

#### core code -- A star algorithm
The code below is the core algorithm we used in the contest----- A* search. The function is generalized enough, the parameters include gameState, problem, and heuristic, we just need simply:
- define different "problem" class
- define a different Heuristic function

```python
  def aStarSearch(self, problem, gameState, heuristic=nullHeuristic):
    from util import PriorityQueue
    start_state = problem.getStartState()
    # store the fringe use priority queue to ensure pop out lowest cost
    fringe = PriorityQueue()
    h = heuristic(start_state, gameState)
    g = 0
    f = g + h
    start_node = (start_state, [], g)
    fringe.push(start_node, f)
    explored = []
    while not fringe.isEmpty():
      current_node = fringe.pop()
      state = current_node[0]
      path = current_node[1]
      current_cost = current_node[2]
      if state not in explored:
        explored.append(state)
        if problem.isGoalState(state):
          return path
        successors = problem.getSuccessors(state)
        for successor in successors:
          current_path = list(path)
          successor_state = successor[0]
          move = successor[1]
          g = successor[2] + current_cost
          h = heuristic(successor_state, gameState)
          if successor_state not in explored:
            current_path.append(move)
            f = g + h
            successor_node = (successor_state, current_path, g)
            fringe.push(successor_node, f)
            # self.debugDraw(successor_node[0], [139, 125, 107],False)
    return []
```

## A star Attacker

In A* Attacker, we defined four different problems:
- search Food Problem: the goal state is food
- Escape Problem: the goal state is home and capsules
- search capsule problem: the goal state is capsules
- search dangerous food problem: the goal state is dangerous food
- back home problem: the goal state is home boudary

The code for these "problem class" can be found in myTeam.py from Line376 to Line483

The Heuristic we used in A start are all the same, which is shown below, we use this heuristic to avoid the ghost.

```python
  def GeneralHeuristic(self, state, gameState):
    heuristic = 0
    if self.getNearestGhostDistance(gameState) != None :
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      ghosts = [a for a in enemies if not a.isPacman and a.scaredTimer < 2 and a.getPosition() != None]
      if ghosts != None and len(ghosts) > 0:
        ghostpositions = [ghost.getPosition() for ghost in ghosts]
        ghostDists = [self.getMazeDistance(state,ghostposition) for ghostposition in ghostpositions]
        ghostDist = min(ghostDists)
        if ghostDist < 2:
          heuristic = pow((5-ghostDist),5)
    return heuristic
```

The decision tree is hand-coded, we use if else to decide which "problem" is passed into A star algorithm. In brief:
- use "search food problem" when ghost position unknown
- use  "escape problem" when a ghost is chasing after us
- use "eat dangerous problem" when a ghost is scared
- use "back home problem" when food left is less than 3
![Astar_Attacker](/uploads/d2efb9f9b6f6813acd6c5e3b02c4dece/Astar_Attacker.PNG)

## A star Defender
In A* Defender, we defined four different problems:
- search last eaten food problem: the goal state is last eaten food
- search invaders problem: the goal state is invader if invader distance is known
- search food problem: the goal state is search food (only used when no invader)
- back home problem: the goal state is home (only used when the defender is outside home and eating food)

The decision tree is hand-coded, we use if else to decide which "problem" is passed into A star algorithm. In brief:
- when the number of invader ==0, use search food problem
- if the number of invader >0 and self not scared
  - if invader position not known: use search last eaten food problem
  - if invader position is known: use search invader problem

in other condition, we use feature and weight to decide the move, we tend to pace up and down in boundary until the invader position or last eaten food position is known.
The detail decision tree is in myTeam.py from Line454 to Line496

![defender_Astar](/uploads/d1f29bbd04c09bcfbfc3ce0baad531d8/defender_Astar.PNG)

# Q-learing agents

__note: the code of q-approximate is in qApproximate.py under pacnman-contest file, the code is not in myTeam.py__


In this project, a Q-Approximate defender is developed but It is not used in the tournament since we find that the performance of the q-approximate is very limited due to the difficulty of defining rewards. 


The most important thing during designing Q-Approximate defender agent is how to define reward. In different conditions, we must have different reward to influence the actions choosed by agent.

Q(S,A)=(1-alpha)*Q(S,A) + alpha*[Reward + gamma*max_aQ(S',a)]
Learn from the training formula of Q-learning. Where α is the learning rate and γ is the discount factor. The greater the learning rate α, the less the effect of pre-retention training. The greater the discount factor γ, the more effective the previous training experience. In the program, we chose different combinations of alpha and gamma. Finally choose "α = 0.01, γ = 1"

For each iteration, the weight value of each feature is changed. After continuous training, find the best stage. At this point, record the weight value and try to compete against other agents to filter out the best QLearning defender.

#### Core code

The code below is the core algorithm we used in the Q-Approximate defender ----- Approximate Q-learning.
```python

---update Qvalue according to formula---

  def update(self, state, action, nextState, reward):
    self.qValues[(state, action)] = (1 - self.alpha) * self.qValues[(state, action)] + self.alpha * (
                reward + self.discount * self.computeValueFromQValues(nextState))
  
---Override---

  def update(self, state, action, nextState, reward):
    r = reward
    actions = nextState.getLegalActions(self.index)
    values = [self.getQvalue(nextState, a) for a in actions]
    maxValue = max(values)
    weights = self.getWeights()
    features = self.getFeatures(state, action)
    print "update_features:"
    print features
    for feature in features:
      print "feature:"
      print feature
      r = self.changeReward(state, r)
      print r
      difference = (r + self.discount * maxValue) - self.getQvalue(state, action)
      print "weights[feature] ", "difference ", "features[feature]"
      print weights[feature], difference, features[feature]
      self.weights[feature] = weights[feature] + self.alpha * difference * features[feature]
      print self.weights[feature]
      print ""
    print "update_weights:"
    print features
    print self.weights
```
#### Q-Approximate Defender
In Q-Approximate Defender, firstly，we defined some different weights:
- numInvaders：The number of invaders
- onDefense: The sttus of pacman
- invaderDistance: The distance between ghost and pacman
- stop: The action "stop"
- reverse: The action "reverse"
- DistToBoundary: Get a positions list of the boundary
- distToLastFood: Get the position of the last eaten food

Before implement learning， we give these weights initial value respectively. Therefore, the learning time has been decreased.
We also defined 3 kinds of rewards for learning：
- When the sacred time is not None，we give a negative reward so that the value of the action that reverses to an invader is large.
- Under the first condition, if our defender could know the distance between itself and invader, the reward will smaller than 
  former's reward.
- In a normal situation, f our defender could know the distance between itself and invader, we give a large positive reward for 
  ensuring the value of action that closes to an invader is large.

The code below is about reward and weights:
```python

---Initialize weighte---

class DefensiveQAgent(QlearningAgent):

  def __init__(self, index, timeForComputing=.1, **args):
    QlearningAgent.__init__(self, index, timeForComputing, **args)
    self.lastEatenFoodPosition = None
    self.filename = "offensive.train.txt"
    if self.numTraining == 0:
      self.epsilon = 0.0 # no exploration
      self.alpha = 0.01 # no learning
      self.weights = self.getInitWeights()

  def getInitWeights(self):
    return util.Counter({'numInvaders': -1000.0,
                         'onDefense': 100.0,
                         'invaderDistance': 500.0,
                         'stop': -100.0,
                         'reverse': -2.0,
                         'DistToBoundary': 1.0,
                         'distToLastFood': 20.0})

---Change Reward---

  def changeReward(self, state, nextState, reward):
   new_reward = reward
   current_state = state.getAgentState(self.index)
   next_state = nextState.getAgentState(self.index)
   current_pos = current_state.getPosition()
   next_pos = next_state.getPosition()
   if next_pos == current_pos:
     new_reward = -9.8765
   if state.getAgentState(self.index).scaredTimer > 0:
     current_dists = self.distToInvader(state)
     if not current_dists is None:
       new_reward = reward
     else:
       new_reward = -9.8765
   else:
     current_dists = self.distToInvader(state)
     print current_dists
     if not current_dists is None:
       new_reward = 9.9875
   return new_reward
```

This below picture shows an example of the weights output:

![weights](/uploads/f2ba56860dbf0c40d27eacb4b548905b/weights.png)

#### Challenges
- Too long trainning time

In the earliest tests, the weight in Q approximate was not initialized, and the value was obtained by the self-training of the Agent. Since there are many long channels in many random maps, the agent can enter the effective range after passing through the channel. Therefore, just the process of training the agent into the effective range takes a long training time. Therefore, we gave the weight initial value so that the agent can quickly reach the valid range.

- Rigorous reward settings

According to the principle of Qlearning, if the next state is the state we want, this state will be set to a reward. The agent selects the action with the largest Qvalue. According to the principle of Qlearning, if the next state is the state we want, this state will be set to a reward. The action selected by the agent is accompanied by the largest Qvalue. However, in the App, we only update the Weights, and the Features will be changed according to the decision tree. Qvalue takes the product of Weights and Features. Therefore, the value of reward may cause the weight updated value to become extreme, and Qvalue will not be what we expected. For example, some weight values may jump between positive and negative numbers, resulting in a deadlock.

- Choose the value of alpha and gamma

In order for the agent to quickly reach the valid position, we give the initial value of the weight. In fact, this is a process of artificial learning. When every time the agent learns, this initial value are changing. Therefore, if the value of alpha is too large, it will cause the weight to change too fast. If the value of gamma is too small, it will cause the agent to despise the experience. Eventually, an action may always be chosen because its Qvalue grows too fast. This effect goes beyond the control of reward and decision trees.

- Generalization

Training with random maps, but the ideal state values obtained each time are not necessarily suitable for other random maps. In some cases, the new Weight will exceed the ideal value, causing the Agent to exhibit unstable and abnormal performance.


## Scan map 
Before the game starts, we have 15 seconds to do the calculation. 
Our team mainly scan the map use BFS (breadth-first search) to determine the safe-food and dangerous-food, as shown in the figure below, the blue dot represents safe food while the red dot represents dangerous food.


The Scan-map method is in myTeam.py from Line 792 to 890


![Scan_map](/uploads/53411efdd6e78cfc1e8cbcf1ed5be6ca/Scan_map.PNG)

## Definition of safe food and dangerous food 
We assume that only the food who has at least two directions back home as safe food, the food that only has one direction back home as dangerous food.
For example, The figure below shows the food that has two directions home, thus, it is defined as safe food.
![20181011_085836335_iOS](/uploads/6dace08d00a5aff9f569d8185a520548/20181011_085836335_iOS.png)

Other food that only has one direction back home as dangerous food. The picture below is an example, the food only has one way back home, and it is then defined as dangerous food.
![20181011_090632179_iOS](/uploads/2e6a729b968252480a59be51fd5e662c/20181011_090632179_iOS.png)

## BFS (breadth-first search)
The basic idea of identifying the dangerous food and safe food is using BFS.
1. if the food is surrounded by three walls, it would be directly defined as dangerous food
2. if the food is surrounded by less then three walls, we can do BFS multiple times, the ways to do BFS (Note: the explanation below might be complicated, you can directly go to myTeam.py from line819 - line917 to see what happened) :
  - get successor position that is not wall
  - for each successor, start BFS, make the successor as the start state
  - before the search, put this successor in closed list, put the position of the food in the closed list
  - make the home boundary as the goal state
  - we define an count if one successor can reach the goal state, count += count
  - if count >1, the food is safe food
