# QapproximateAgents.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
from util import nearestPoint
import game
import copy


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='', second='DefensiveQAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class QapproximateAgents(CaptureAgent):

  def __init__(self, index, timeForComputing=.1, numTraining=0, epsilon=0.5, alpha=0.1, gamma=1, **args):
    CaptureAgent.__init__(self, index, timeForComputing)
    self.episodesSoFar = 0
    self.accumTrainRewards = 0.0
    self.accumTestRewards = 0.0
    self.numTraining = int(numTraining)
    self.epsilon = float(epsilon)
    self.alpha = float(alpha)
    self.discount = float(gamma)
    self.qValues = util.Counter()

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.start = gameState.getAgentPosition(self.index)
    self.midWidth = gameState.data.layout.width / 2
    self.height = gameState.data.layout.height
    self.width = gameState.data.layout.width
    self.midHeight = gameState.data.layout.height / 2
    self.foodEaten = 0
    self.initialnumberOfFood = len(self.getFood(gameState).asList())
    self.lastEatenFoodPosition = None
    self.initialnumberOfCapsule = len(self.getCapsules(gameState))
    CaptureAgent.registerInitialState(self, gameState)
    self.startEpisode()
    if self.episodesSoFar == 0:
      print 'Beginning %d episodes of Training' % (self.numTraining)

  def startEpisode(self):
    self.lastState = None
    self.lastAction = None
    self.episodeRewards = 0.0
    print self.getWeights()

  def stopEpisode(self):
    if self.episodesSoFar < self.numTraining:
      self.accumTrainRewards += self.episodeRewards
    else:
      self.accumTestRewards += self.episodeRewards
    self.episodesSoFar += 1
    if self.episodesSoFar >= self.numTraining:
      # Take off the training wheels
      self.epsilon = 0.0  # no exploration
      self.alpha = 0.0  # no learning

  def isInTraining(self):
    return self.episodesSoFar < self.numTraining

  def isInTesting(self):
    return not self.isInTraining()

  def getQvalue(self, gameState, action):
    return self.qValues[(gameState, action)]

  def computeValueFromQValues(self, gameState):
    actions = gameState.getLegalActions(self.index)
    if len(actions) == 0:
      return 0.0
    values = [self.getQvalue(gameState, a) for a in actions]
    return max(values)

  def computeActionFromQValues(self, gameState):
    actions = gameState.getLegalActions(self.index)
    values = [self.getQvalue(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    if len(bestActions) == 0:
      return None
    return random.choice(bestActions)

  def update(self, state, action, nextState, reward):
    self.qValues[(state, action)] = (1 - self.alpha) * self.qValues[(state, action)] + self.alpha * (
                reward + self.discount * self.computeValueFromQValues(nextState))

  def observationFunction(self, state):
    if not self.lastState is None:
      reward = (state.getScore() - self.lastState.getScore())
      self.observeTransition(self.lastState, self.lastAction, state, reward)
    return CaptureAgent.observationFunction(self, state)

  def observeTransition(self, state, action, nextState, deltaReward):
    self.episodeRewards += deltaReward
    self.update(state, action, nextState, deltaReward)

  def doAction(self, state, action):
    self.lastState = state
    self.lastAction = action

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    action = None
    if util.flipCoin(self.epsilon):
      action = random.choice(actions)
    else:
      action = self.computeActionFromQValues(gameState)
      print action
    self.doAction(gameState, action)  # from Q learning agent
    return action

  def final(self, state):
    CaptureAgent.final(self, state)
    deltaReward = state.getScore() - self.lastState.getScore()
    print state.getScore(), self.lastState.getScore()
    self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
    self.stopEpisode()

    # Make sure we have this var
    if not 'episodeStartTime' in self.__dict__:
      self.episodeStartTime = time.time()
    if not 'lastWindowAccumRewards' in self.__dict__:
      self.lastWindowAccumRewards = 0.0
    self.lastWindowAccumRewards += state.getScore()

    NUM_EPS_UPDATE = 100
    if self.episodesSoFar % NUM_EPS_UPDATE == 0:
      print 'Reinforcement Learning Status:'
      windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
      if self.episodesSoFar <= self.numTraining:
        trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
        print '\tCompleted %d out of %d training episodes' % (
          self.episodesSoFar, self.numTraining)
        print '\tAverage Rewards over all training: %.2f' % (
          trainAvg)
      else:
        testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
        print '\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining)
        print '\tAverage Rewards over testing: %.2f' % testAvg
      print '\tAverage Rewards for last %d episodes: %.2f' % (
        NUM_EPS_UPDATE, windowAvg)
      print '\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime)
      self.lastWindowAccumRewards = 0.0
      self.episodeStartTime = time.time()

    if self.episodesSoFar == self.numTraining:
      msg = 'Training Done (turning off epsilon and alpha)'
      print '%s\n%s' % (msg, '-' * len(msg))
      print self.getWeights()


#######################
#   DefensiveQAgent   #
#######################


class DefensiveQAgent(QapproximateAgents):

  def __init__(self, index, timeForComputing=.1, **args):
    QapproximateAgents.__init__(self, index, timeForComputing, **args)
    self.lastEatenFoodPosition = None
    self.filename = "offensive.train.txt"

    # # initialize weights
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

  def loadWeights(self, filename):
    with open(filename) as f:
      self.weights = eval(f.readline())

  def saveWeights(self, filename):
    with open(filename, "w") as f:
      f.write(str(self.weights))

  def distToInvader(self, state):
    myState = state.getAgentState(self.index)
    myPos = myState.getPosition()
    enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
    else:
      dists = None
    return dists

  def distToHome(self, gameState):
    myState = gameState.getAgentState(self.index)
    myPosition = myState.getPosition()
    boundaries = []
    i = self.midWidth - 1
    boudaries = [(i,j) for j in  range(self.height)]
    validPositions = []
    for i in boudaries:
      if not gameState.hasWall(i[0],i[1]):
        validPositions.append(i)
    dist = 9999
    for validPosition in validPositions:
      tempDist =  self.getMazeDistance(validPosition,myPosition)
      if tempDist < dist:
        dist = tempDist
        temp = validPosition
    return dist

  def locationOfLastEatenFood(self, gameState):
    if len(self.observationHistory) > 1:
      prevState = self.getPreviousObservation()
      prevFoodList = self.getFoodYouAreDefending(prevState).asList()
      currentFoodList = self.getFoodYouAreDefending(gameState).asList()
      if len(prevFoodList) != len(currentFoodList):
        for food in prevFoodList:
          if food not in currentFoodList:
            self.lastEatenFoodPosition = food

  def observeTransition(self, lastSate, lastAction, nextState, deltaReward):
    self.episodeRewards += deltaReward
    self.update(lastSate, lastAction, nextState, deltaReward)

  def doAction(self, gameState, action):
    self.lastState = gameState
    self.lastAction = action

  def observationFunction(self, state):
    if not self.lastState is None:
      reward = 0.0
      # reward -= 1.
      self.observeTransition(self.lastState, self.lastAction, state, reward)
    return CaptureAgent.observationFunction(self, state)

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    action = None
    if util.flipCoin(self.epsilon):
      action = random.choice(actions)
    else:
      action = self.computeActionFromQValues(gameState)
    self.doAction(gameState, action)  # from Q learning agent
    return action

  def computeActionFromQValues(self, gameState):
    actions = gameState.getLegalActions(self.index)
    values = [self.getQvalue(gameState, a) for a in actions]
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman]
    knowninvaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    if len(invaders) == 0 or gameState.getAgentPosition(self.index) == self.lastEatenFoodPosition or len(knowninvaders) > 0:
      self.lastEatenFoodPosition = None
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def getSuccessor(self, gameState,  action):
    """Finds the next successor which is a grid position (location tuple)."""
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def getWeights(self):
    return self.weights

  def getQvalue(self, gameState, action):
      weights = self.getWeights()
      features = self.getFeatures(gameState, action)
      return weights * features

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    features['invaderDistance'] = 0
    features['stop'] = 0.0
    features['onDefense'] = 1.0
    features['reverse'] = 0.0
    features['distToLastFood'] = 0.0

    # Computes whether we're on defense (1) or offense (0)
    if myState.isPacman: features['onDefense'] = 0.0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = float(len(invaders))
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = float(1/min(dists))
      if gameState.getAgentState(self.index).scaredTimer > 0:
        features['invaderDistance'] = float(-1/min(dists))
    if action == Directions.STOP: features['stop'] = 1.0
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

    # Computes distance to Boundary
    if action == rev: features['reverse'] = 1.0
    features['DistToBoundary'] = float(- self.distToHome(successor))
    self.locationOfLastEatenFood(gameState)

    if not self.lastEatenFoodPosition is None:
      self.debugDraw(self.lastEatenFoodPosition, [139, 125, 107], True)
      features['distToLastFood'] = float(- self.getMazeDistance(myPos, self.lastEatenFoodPosition))
      features['DistToBoundary'] = 0.0
    return features

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

  def update(self, state, action, nextState, reward):
    actions = nextState.getLegalActions(self.index)
    values = [self.getQvalue(nextState, a) for a in actions]
    maxValue = max(values)
    weights = self.getWeights()
    features = self.getFeatures(state, action)
    print "update_features:"
    print features
    for feature in features:
      r = self.changeReward(state, nextState, reward)
      difference = (r + self.discount * maxValue) - self.getQvalue(state, action)
      self.weights[feature] = weights[feature] + self.alpha * difference * features[feature]
    print "update_weights:"
    print self.weights

  def final(self, state):
    QapproximateAgents.final(self, state)
    if self.episodesSoFar == self.numTraining:
      filename = self.filename
      self.saveWeights(filename)
      pass