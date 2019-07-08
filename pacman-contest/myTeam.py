# myTeam.py
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

"""
@author: Chenyang Lu
         Pengcheng Yao
         Jing Du
@Date: 13 October, 2018
"""
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from game import Actions
import copy


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'AstarAttacker', second = 'AstarDefender'):
  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  This is the ancestor class for the agent we used.
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.midWidth = gameState.data.layout.width / 2
    self.height = gameState.data.layout.height
    self.width = gameState.data.layout.width
    self.midHeight = gameState.data.layout.height / 2
    self.foodEaten = 0
    self.initialnumberOfFood = len(self.getFood(gameState).asList())
    self.lastEatenFoodPosition = None
    self.initialnumberOfCapsule = len(self.getCapsules(gameState))
    scanmap = ScanMap(gameState, self)
    foodList = scanmap.getFoodList(gameState)

    self.safeFoods = scanmap.getSafeFoods(foodList)    #a list of tuple contains safe food location
    self.dangerFoods = scanmap.getDangerFoods(self.safeFoods)
    # for food in self.safeFoods:
    #   self.debugDraw(food, [100, 100, 255], False)
    # for food in self.dangerFoods:
    #   self.debugDraw(food, [255, 100, 100], False)

    self.blueRebornHeight = self.height -1
    self.blueRebornWidth = self.width -1


  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor (Game state object)
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def chooseAction(self, gameState):
    self.locationOfLastEatenFood(gameState)  # detect last eaten food
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def distToFood(self, gameState):
    ''''
    get the nearestFood, remember to edit again, return  [nearestFood, nearestDistance]
    '''
    food = self.getFood(gameState).asList()
    if len(food) > 0:
      dist = 9999
      for a in food:
        tempDist = self.getMazeDistance(gameState.getAgentPosition(self.index), a)
        if tempDist < dist:
          dist = tempDist
          temp = a
      return dist
    else:
      return 0

  def distToHome(self, gameState):
    ''''
    return the distance to nearest boudndary
    '''
    myState = gameState.getAgentState(self.index)
    myPosition = myState.getPosition()
    boundaries = []
    if self.red:
      i = self.midWidth - 1
    else:
      i = self.midWidth + 1
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


  def boundaryPosition(self,gameState):
    ''''
    return a list of positions of boundary
    '''
    myState = gameState.getAgentState(self.index)
    myPosition = myState.getPosition()
    boundaries = []
    if self.red:
      i = self.midWidth - 1
    else:
      i = self.midWidth + 1
    boudaries = [(i,j) for j in  range(self.height)]
    validPositions = []
    for i in boudaries:
      if not gameState.hasWall(i[0],i[1]):
        validPositions.append(i)
    return validPositions


  def distToCapsule(self,gameState):
    ''''
    return the nerest distance to capsule
    '''
    if len(self.getCapsules(gameState)) > 1:
      dist = 9999
      for i in self.getCapsules(gameState):
        tempDist = self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), i)
        if tempDist < dist:
          dist = tempDist
          self.debugDraw(i, [125, 125, 211], True)
      return dist

    elif len(self.getCapsules(gameState)) == 1 :
      distToCapsule = self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), self.getCapsules(gameState)[0])
      self.debugDraw(self.getCapsules(gameState)[0], [125, 125, 211], True)
      return distToCapsule


  def locationOfLastEatenFood(self,gameState):
    ''''
    return the location of the last eaten food
    '''
    if len(self.observationHistory) > 1:
      prevState = self.getPreviousObservation()
      prevFoodList = self.getFoodYouAreDefending(prevState).asList()
      currentFoodList = self.getFoodYouAreDefending(gameState).asList()
      if len(prevFoodList) != len(currentFoodList):
        for food in prevFoodList:
          if food not in currentFoodList:
            self.lastEatenFoodPosition = food


  def getNearestGhostDistance(self, gameState):
    ''''
    return the distance of the nearest ghost
    '''
    myPosition =  gameState.getAgentState(self.index).getPosition()
    enemies =  [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      dists = [self.getMazeDistance(myPosition, a.getPosition()) for a in ghosts]
      return min(dists)
    else:
      return None


  def getNearestinvader(self, gameState):
    ''''
    return the distance of the nearest ghost
    '''
    myPosition =  gameState.getAgentState(self.index).getPosition()
    enemies =  [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPosition, a.getPosition()) for a in invaders]
      return min(dists)
    else:
      return None



  def DistToGhost(self, gameState):
    ''''
    return a list list[0] is distance to nearest ghost list[1] is the state of ghost
    '''
    myPosition =  gameState.getAgentState(self.index).getPosition()
    enemies =  [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      dist = 999
      for a in ghosts:
        temp = self.getMazeDistance(myPosition, a.getPosition())
        if temp < dist:
          dist = temp
          ghostState = a
      return [dist,ghostState]
    else:
      return None


  def opponentscaredTime(self,gameState):
    opponents = self.getOpponents(gameState)
    for opponent in opponents:
      if gameState.getAgentState(opponent).scaredTimer > 1:
        return gameState.getAgentState(opponent).scaredTimer

    return None


  def nullHeuristic(self,state, problem=None):
    return 0

  """
  A Genral start search that can be used to solve any problem and any heuristics 
  """

  def aStarSearch(self, problem, gameState, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
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
    return []



  def GeneralHeuristic(self, state, gameState):

    """

    This heuristic is used for to avoid ghoost, we give the
    position which close to ghost a higher heuristic to avoid
    colission with ghost

    """
    heuristic = 0
    if self.getNearestGhostDistance(gameState) != None :
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      # pacmans = [a for a in enemies if a.isPacman]
      ghosts = [a for a in enemies if not a.isPacman and a.scaredTimer < 2 and a.getPosition() != None]
      if ghosts != None and len(ghosts) > 0:
        ghostpositions = [ghost.getPosition() for ghost in ghosts]
        # pacmanPositions = [pacman.getPosition() for pacman in pacmans]
        ghostDists = [self.getMazeDistance(state,ghostposition) for ghostposition in ghostpositions]
        ghostDist = min(ghostDists)
        if ghostDist < 2:
          # print ghostDist
          heuristic = pow((5-ghostDist),5)

    return heuristic

  """

  This heuristic is used for to avoid pacman when the ghoost is scared, we give the 
  position which close to ghost a higher heuristic to avoid 
  colission with ghost

  """

  def avoidPacmanHeuristic(self, state, gameState):
    weight = 0
    if self.getNearestinvader(gameState) != None and gameState.getAgentState(self.index).scaredTimer > 0:
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      pacmans = [a for a in enemies if a.isPacman and a.getPosition() != None]
      if pacmans != None and len(pacmans) > 0:
        pacmanPositions = [pacman.getPosition() for pacman in pacmans]
        pacmanDists = [self.getMazeDistance(state,pacmanposition) for pacmanposition in pacmanPositions]
        pacmanDist = min(pacmanDists)
        if pacmanDist < 2:
          # print ghostDist
          weight = pow((5- pacmanDist),5)

    return weight

###########################################
#            Astar Attacker               #
###########################################
  """

  Below is the a* attacker agent

  """

class AstarAttacker(DummyAgent):

  def getGhostDistance(self,gameState,index):
    myPosition = gameState.getAgentState(self.index).getPosition()
    ghost = gameState.getAgentState(index)
    dist = self.getMazeDistance(myPosition,ghost.getPosition())
    return dist


  def chooseAction(self, gameState):
    myState = gameState.getAgentState(self.index)
    myPosition = myState.getPosition()
    newSafeFoods =[]
    newDangerousFood =[]

    """
    update  safe food and dangerous food
    """
    for i in self.getFood(gameState).asList():
      if i in self.safeFoods:
        newSafeFoods.append(i)

    for i in self.getFood(gameState).asList():
      if i in self.dangerFoods:
        newDangerousFood.append(i)

    self.safeFoods = copy.deepcopy(newSafeFoods)
    self.dangerFoods = copy.deepcopy(newDangerousFood)


    """
    Decision Tree
    """
    if gameState.getAgentState(self.index).numCarrying == 0 and len(self.getFood(gameState).asList()) == 0:
      return 'Stop'

    if len(self.safeFoods) < 1 and len(self.getCapsules(gameState)) != 0 and self.opponentscaredTime(gameState) < 10:
      problem = SearchCapsule(gameState, self, self.index)
      return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]


    if gameState.getAgentState(self.index).numCarrying < 1 and (len(self.safeFoods) > 0):
      problem = SearchSafeFood(gameState, self, self.index)
      return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

    if gameState.getAgentState(self.index).numCarrying < 1 and (len(self.safeFoods) == 0):
      problem = SearchFood(gameState, self, self.index)
      return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

    if self.DistToGhost(gameState) != None and self.DistToGhost(gameState)[0]< 6 and \
        self.DistToGhost(gameState)[1].scaredTimer < 5:
      problem = Escape(gameState, self, self.index)
      if len(self.aStarSearch(problem, self.GeneralHeuristic)) == 0:
        return 'Stop'
      else:
        return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

    if self.opponentscaredTime(gameState) != None:
      if self.opponentscaredTime(gameState) > 20 and len(self.dangerFoods) > 0:
        problem = SearchDangerousFood(gameState, self, self.index)
        return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

    if len(self.getFood(gameState).asList()) < 3 or gameState.data.timeleft < self.distToHome(gameState) + 60\
        or gameState.getAgentState(self.index).numCarrying > 15:
      problem = BackHome(gameState, self, self.index)
      if len(self.aStarSearch(problem, self.GeneralHeuristic)) == 0:
        return 'Stop'
      else:
        return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

    problem = SearchFood(gameState, self, self.index)
    return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]



#######################
#  Astart_defender   #
#######################

class AstarDefender(DummyAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def chooseAction(self, gameState):

    myState = gameState.getAgentState(self.index)
    myPosition = myState.getPosition()
    self.locationOfLastEatenFood(gameState)  # detect last eaten food
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman]
    knowninvaders = [a for a in enemies if a.isPacman and a.getPosition() !=None ]

    if len(invaders) == 0 or gameState.getAgentPosition(self.index) == self.lastEatenFoodPosition or len(knowninvaders) > 0:
      self.lastEatenFoodPosition = None

    # if number of invaders is less than two, we can go out and try to eat some food
    if len(invaders) < 1:
      # eat maximum 3 food
      if gameState.getAgentState(self.index).numCarrying < 3 and len(self.getFood(gameState).asList()) != 0 and not (self.DistToGhost(gameState) != None and self.DistToGhost(gameState)[0]< 4 and \
        self.DistToGhost(gameState)[1].scaredTimer < 2):
        problem = SearchFood(gameState, self, self.index)
        return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]
      else:
        problem = BackHome(gameState, self, self.index)
        if len(self.aStarSearch(problem, self.GeneralHeuristic)) == 0:
          return 'Stop'
        else:
          return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

     # when number of invader > 0, we excute defendense strategy
    else:

      if len(knowninvaders) == 0 and  self.lastEatenFoodPosition!=None and gameState.getAgentState(self.index).scaredTimer == 0:
        problem = SearchLastEatenFood(gameState,self,self.index)
        return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

      # chase the invader only the distance is Known and ghost not scared
      if len(knowninvaders) > 0 and gameState.getAgentState(self.index).scaredTimer == 0:
        problem =  SearchInvaders(gameState,self,self.index)
        return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    features['dead'] = 0

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0 and gameState.getAgentState(self.index).scaredTimer >0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = -1/min(dists)


    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    features['DistToBoundary'] = - self.distToHome(successor)
    return features

    if gameState.isOnRedTeam(self.index):
      if gameState.getAgentState(self.index) == (1,1):
        features['dead'] = 1
    else:
      if gameState.getAgentState(self.index) == (self.height-1,self.width-1):
        features['dead'] = 1


  def getWeights(self, gameState, action):
    return {'invaderDistance':1000,'onDefense': 200, 'stop': -100, 'reverse': -2,'DistToBoundary': 1,'dead':-10000}


##############################################################
#  Helper Class  ----- defines multiple  Search problem      #
##############################################################


class PositionSearchProblem:
    """
    It is the ancestor class for all the search problem class.
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point.
    """

    def __init__(self, gameState, agent, agentIndex = 0,costFn = lambda x: 1):
        self.walls = gameState.getWalls()
        self.costFn = costFn
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
      return self.startState

    def isGoalState(self, state):

      util.raiseNotDefined()

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost


class SearchFood(PositionSearchProblem):
  """
   The goal state is to find all the food
  """

  def __init__(self, gameState, agent, agentIndex = 0):
    "Stores information from the gameState.  You don't need to change this."
    # Store the food for later reference
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    # Store info for the PositionSearchProblem (no need to change this)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
    self.carry = gameState.getAgentState(agentIndex).numCarrying
    self.foodLeft = len(self.food.asList())


  def isGoalState(self, state):
    # the goal state is the position of food or capsule
    # return state in self.food.asList() or state in self.capsule
    return state in self.food.asList()

class SearchSafeFood(PositionSearchProblem):
  """
  The goal state is to find all the safe fooof
  """

  def __init__(self, gameState, agent, agentIndex = 0):
    "Stores information from the gameState.  You don't need to change this."
    # Store the food for later reference
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    # Store info for the PositionSearchProblem (no need to change this)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
    self.carry = gameState.getAgentState(agentIndex).numCarrying
    self.foodLeft = len(self.food.asList())
    self.safeFood = agent.safeFoods


  def isGoalState(self, state):
    # the goal state is the position of food or capsule
    # return state in self.food.asList() or state in self.capsule
    return state in self.safeFood

class SearchDangerousFood(PositionSearchProblem):
  """
  Used to get the safe food
  """

  def __init__(self, gameState, agent, agentIndex = 0):
    "Stores information from the gameState.  You don't need to change this."
    # Store the food for later reference
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    # Store info for the PositionSearchProblem (no need to change this)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
    self.carry = gameState.getAgentState(agentIndex).numCarrying
    self.foodLeft = len(self.food.asList())
    self.dangerousFood = agent.dangerFoods

  def isGoalState(self, state):
    # the goal state is the position of food or capsule
    # return state in self.food.asList() or state in self.capsule
    return state in self.dangerousFood


class Escape(PositionSearchProblem):
  """
  Used to escape
  """

  def __init__(self, gameState, agent, agentIndex=0):
    "Stores information from the gameState.  You don't need to change this."
    # Store the food for later reference
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    # Store info for the PositionSearchProblem (no need to change this)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
    self.homeBoundary = agent.boundaryPosition(gameState)
    self.safeFood = agent.safeFoods

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
    # the goal state is the boudary of home or the positon of capsule
    return state in self.homeBoundary or state in self.capsule


class BackHome(PositionSearchProblem):
  """
  Used to go back home
  """

  def __init__(self, gameState, agent, agentIndex=0):
    "Stores information from the gameState.  You don't need to change this."
    # Store the food for later reference
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    # Store info for the PositionSearchProblem (no need to change this)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
    self.homeBoundary = agent.boundaryPosition(gameState)

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
    # the goal state is the boudary of home or the positon of capsule
    return state in self.homeBoundary


class SearchCapsule(PositionSearchProblem):
  """
  Used to search capsule
  """

  def __init__(self, gameState, agent, agentIndex = 0):
    # Store the food for later reference
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    # Store info for the PositionSearchProblem (no need to change this)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE


  def isGoalState(self, state):
    # the goal state is the location of capsule
    return state in self.capsule


class SearchLastEatenFood(PositionSearchProblem):
  """
  Used to search capsule
  """

  def __init__(self, gameState, agent, agentIndex = 0):
    # Store the food for later reference
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    # Store info for the PositionSearchProblem (no need to change this)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.lastEatenFood = agent.lastEatenFoodPosition
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

  def isGoalState(self, state):
    # the goal state is the location of capsule
    return state == self.lastEatenFood



class SearchInvaders(PositionSearchProblem):
  """
  Used to search capsule
  """

  def __init__(self, gameState, agent, agentIndex = 0):
    # Store the food for later reference
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    # Store info for the PositionSearchProblem (no need to change this)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.lastEatenFood = agent.lastEatenFoodPosition
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
    self.enemies = [gameState.getAgentState(agentIndex) for agentIndex in agent.getOpponents(gameState)]
    self.invaders = [a for a in self.enemies if a.isPacman and a.getPosition != None]
    if len(self.invaders) > 0:
      self.invadersPosition =  [invader.getPosition() for invader in self.invaders]
    else:
      self.invadersPosition = None

  def isGoalState(self, state):
    # # the goal state is the location of invader
    return state in self.invadersPosition


##########################
#    Scan Map Method     #
##########################


class ScanMap:

  """
  A Class Below is used for scanning the map to find
  Safe food and dangerousfood

  Note: Safe food is the food whitin the position can has
  at least two ways home
  """


  def __init__(self, gameState, agent):
      "Stores information from the gameState.  You don't need to change this."
      # Store the food for later reference
      self.food = agent.getFood(gameState).asList()
      # Store info for the PositionSearchProblem (no need to change this)
      self.walls = gameState.getWalls()
      self.homeBoundary = agent.boundaryPosition(gameState)
      self.height = gameState.data.layout.height
      self.width = gameState.data.layout.width

  def getFoodList(self, gameState):
      foods = []
      for food in self.food:
          food_fringes = []
          food_valid_fringes = []
          count = 0
          food_fringes.append((food[0] + 1, food[1]))
          food_fringes.append((food[0] - 1, food[1]))
          food_fringes.append((food[0], food[1] + 1))
          food_fringes.append((food[0], food[1] - 1))
          for food_fringe in food_fringes:
              if not gameState.hasWall(food_fringe[0], food_fringe[1]):
                  count = count + 1
                  food_valid_fringes.append(food_fringe)
          if count > 1:
              foods.append((food, food_valid_fringes))
      return foods

  def getSafeFoods(self, foods):
      safe_foods = []
      for food in foods:
          count = self.getNumOfValidActions(food)
          if count > 1:
              safe_foods.append(food[0])
      return safe_foods

  def getDangerFoods(self, safe_foods):
      danger_foods = []
      for food in self.food:
          if food not in safe_foods:
              danger_foods.append(food)
      return danger_foods

  def getSuccessors(self, state):
      successors = []
      for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
          x, y = state
          dx, dy = Actions.directionToVector(action)
          nextx, nexty = int(x + dx), int(y + dy)
          if not self.walls[nextx][nexty]:
              nextState = (nextx, nexty)
              successors.append((nextState, action))
      return successors

  def isGoalState(self, state):
      return state in self.homeBoundary

  def getNumOfValidActions(self, foods):
      food = foods[0]
      food_fringes = foods[1]
      visited = []
      visited.append(food)
      count = 0
      for food_fringe in food_fringes:
          closed = copy.deepcopy(visited)
          if self.BFS(food_fringe, closed):
              count = count + 1
      return count

  def BFS(self, food_fringe, closed):
      from util import Queue

      fringe = Queue()
      fringe.push((food_fringe, []))
      while not fringe.isEmpty():
          state, actions = fringe.pop()
          closed.append(state)
          if self.isGoalState(state):
              return True
          for successor, direction in self.getSuccessors(state):
              if successor not in closed:
                  closed.append(successor)
                  fringe.push((successor, actions + [direction]))