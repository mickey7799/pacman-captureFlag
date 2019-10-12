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


import operator
import random
import time

import game
import util
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint

START_MODE = 'start'
ATTACK_MODE = 'attack'
DEFEND_MODE = 'defend'
HUNT_MODE = 'hunt'

POWERCAPSULETIME = 120

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='TopAgent', second='BottomAgent'):
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


class GlobalState:
  defendingAgent = -1
  observations = None
  deadEndPositions = None
  dangerPositionDictionary = None
  potentialTargetFoods = {}

  @staticmethod
  def calculateDeadEndPositions(gameState):
    GlobalState.deadEndPositions = []
    width = gameState.getWalls().width
    height = gameState.getWalls().height
    walls = gameState.getWalls().asList()

    for i in range(0, width):
      for j in range(0, height):
        numWallAround = 0

        if (i, j) not in walls:
          if (i - 1, j) in walls:
            numWallAround += 1
          if (i + 1, j) in walls:
            numWallAround += 1
          if (i, j - 1) in walls:
            numWallAround += 1
          if (i, j + 1) in walls:
            numWallAround += 1
          if numWallAround == 3:
            GlobalState.deadEndPositions.append((i, j))

  @staticmethod
  def calculateDangerPositionDictionary(gameState):
    GlobalState.dangerPositionDictionary = {}

    for deadEndPosition in GlobalState.deadEndPositions:
      GlobalState.calculateDanger(deadEndPosition, gameState.getWalls())

  @staticmethod
  def calculateDanger(deadEndPosition, walls):
    decreasingDangerLevel = 10
    currentDangerLevel = -1000
    currentPosition = deadEndPosition

    if currentPosition in GlobalState.dangerPositionDictionary:
      GlobalState.dangerPositionDictionary[currentPosition] = min(
          GlobalState.dangerPositionDictionary[currentPosition], currentDangerLevel
      )
    else:
      GlobalState.dangerPositionDictionary[currentPosition] = currentDangerLevel

    while True:
      wallNumber = 0
      (currentX, currentY) = currentPosition
      upPosition = (currentX, currentY - 1)
      isUpPositionWall = walls[currentX][currentY - 1]
      downPosition = (currentX, currentY + 1)
      isDownPositionWall = walls[currentX][currentY + 1]
      leftPosition = (currentX - 1, currentY)
      isLeftPositionWall = walls[currentX - 1][currentY]
      rightPosition = (currentX + 1, currentY)
      isRightPositionWall = walls[currentX + 1][currentY]

      layoutPosition = currentPosition
      if upPosition in GlobalState.dangerPositionDictionary or isUpPositionWall == True:
        wallNumber += 1
      else:
        layoutPosition = upPosition
      if downPosition in GlobalState.dangerPositionDictionary or isDownPositionWall == True:
        wallNumber += 1
      else:
        layoutPosition = downPosition
      if leftPosition in GlobalState.dangerPositionDictionary or isLeftPositionWall == True:
        wallNumber += 1
      else:
        layoutPosition = leftPosition
      if rightPosition in GlobalState.dangerPositionDictionary or isRightPositionWall == True:
        wallNumber += 1
      else:
        layoutPosition = rightPosition

      if wallNumber == 1 or wallNumber == 2:
        break
      elif wallNumber == 3:
        currentDangerLevel = currentDangerLevel + decreasingDangerLevel
        GlobalState.dangerPositionDictionary[currentPosition] = currentDangerLevel
        currentPosition = layoutPosition
      else:
        GlobalState.dangerPositionDictionary[currentPosition] = -1000

  @staticmethod
  def getDangerLevel(position):
    dangerLevel = 0

    if position in GlobalState.dangerPositionDictionary:
      dangerLevel = GlobalState.dangerPositionDictionary[position]

    return dangerLevel


class MainAgent(CaptureAgent):

  # Give each agent a most likely position and a power timer
  def __init__(self, gameState):
    CaptureAgent.__init__(self, gameState)
    self.potentialPossition = [None] * 4
    self.holdFoodNumber = 0
    self.hasReachedBorder = False

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on). 

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """
    ''' 
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py. 
    '''
    CaptureAgent.registerInitialState(self, gameState)

    ''' 
    Your initialization code goes here, if you need any.
    '''
    self.isTopAgent = self.index == max(gameState.getRedTeamIndices()
                                        ) or self.index == max(gameState.getBlueTeamIndices())

    # Initialize GlobalState
    if GlobalState.deadEndPositions == None:
      GlobalState.calculateDeadEndPositions(gameState)
      GlobalState.calculateDangerPositionDictionary(gameState)

    # Sets if agent is on red team or not
    if self.red:
      CaptureAgent.registerTeam(self, gameState.getRedTeamIndices())
    else:
      CaptureAgent.registerTeam(self, gameState.getBlueTeamIndices())

    self.opponents = self.getOpponents(gameState)
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.borderGrids = self.getBorderGrids(gameState)
    self.isFirstAgent = self.index == self.agentsOnTeam[0]

    # observations is used to infere the position of enemy agents using noisy data
    if GlobalState.observations == None:
      GlobalState.observations = [util.Counter()] * gameState.getNumAgents()

      # All observations begin with the agent at its initial position
      for agent, _ in enumerate(GlobalState.observations):
        if agent in self.opponents:
          GlobalState.observations[agent][gameState.getInitialAgentPosition(agent)] = 1

    self.start(gameState)

  def getBorderGrids(self, gameState):
    homeBorderGrids = []
    width = gameState.getWalls().width
    height = gameState.getWalls().height
    (startX, _) = self.getCurrentPosition(gameState)

    # blue (right) and red (left)
    # example :
    # layout width : 32
    # postion x 0~15 is blue team , 16~31 is red team
    # 32 / 2 - 1 = 15 is the blue team border position
    # 32 / 2 = 16 is the red team border position
    if self.red:
      positionX = int(width / 2 - 1)
    else:
      positionX = int(width / 2)

    for i in range(0, height):
      if gameState.getWalls()[positionX][i] != True:  # avoid border position is wall position
        homeBorderGrids.append((positionX, i))

    return homeBorderGrids

  # Detect position of enemies that are visible
  def getGhostPositions(self, gameState):
    enemyPositions = []

    for enemy in self.getOpponents(gameState):
      position = gameState.getAgentPosition(enemy)
      isPacman = gameState.getAgentState(enemy).isPacman

      if position != None and not isPacman:
        enemyPositions.append((enemy, position))

    return enemyPositions

  # Detect position of invader that are visible
  def getInvaderPositions(self, gameState):
    enemyPositions = []

    for enemy in self.getOpponents(gameState):
      position = gameState.getAgentPosition(enemy)
      isPacman = gameState.getAgentState(enemy).isPacman

      if position != None and isPacman:
        enemyPositions.append((enemy, position))

    return enemyPositions

  # Find the closest visible ghost
  def getMinGhostDistance(self, gameState):
    positions = self.getGhostPositions(gameState)
    minDistance = -1

    if len(positions) > 0:
      minDistance = float('inf')
      currentPosition = gameState.getAgentPosition(self.index)

      for _, position in positions:
        minDistance = min(minDistance, self.getMazeDistance(currentPosition, position))

    return minDistance

  # Find the closest visible invader
  def getMinInvaderDistance(self, gameState):
    positions = self.getInvaderPositions(gameState)
    minDistance = 0

    if len(positions) > 0:
      minDistance = float('inf')
      currentPosition = gameState.getAgentPosition(self.index)

      for _, position in positions:
        minDistance = min(minDistance, self.getMazeDistance(currentPosition, position))

    return minDistance

  def getMinScaredGhostDistance(self, gameState):
    scaredGhostPositoins = []

    for enemy in self.getOpponents(gameState):
      position = gameState.getAgentPosition(enemy)
      isPacman = gameState.getAgentState(enemy).isPacman
      isScared = gameState.getAgentState(enemy).scaredTimer != 0

      if position != None and not isPacman and isScared:
        scaredGhostPositoins.append((enemy, position))

    minDistance = -1

    if len(scaredGhostPositoins) > 0:
      minDistance = float('inf')
      currentPosition = gameState.getAgentPosition(self.index)

      for _, position in scaredGhostPositoins:
        minDistance = min(minDistance, self.getMazeDistance(currentPosition, position))

    return minDistance

  # Find the closest visible invader
  def getMinInvaderDistanceByAgentIndex(self, gameState, agentIndex):
    positions = self.getInvaderPositions(gameState)
    minDistance = 0

    if len(positions) > 0:
      minDistance = float('inf')
      currentPosition = gameState.getAgentPosition(agentIndex)

      for i, position in positions:
        minDistance = min(minDistance, self.getMazeDistance(currentPosition, position))

    return minDistance

  def isInEnemyTerritory(self, gameState):
    return gameState.getAgentState(self.index).isPacman

  def getCurrentPosition(self, gameState):
    return gameState.getAgentState(self.index).getPosition()

  # Calculates the distance to the partner of the current agent
  def getPartnerDistance(self, gameState):
    distanceToAgent = 0

    if self.isFirstAgent:
      currentPosition = self.getCurrentPosition(gameState)
      otherPosition = gameState.getAgentState(self.agentsOnTeam[1]).getPosition()
      distanceToAgent = self.getMazeDistance(currentPosition, otherPosition)

    return distanceToAgent

  # Which side of the board is the agent?
  def side(self, gameState):
    width, height = gameState.data.layout.width, gameState.data.layout.height
    position = gameState.getAgentPosition(self.index)

    if self.red:
      if position[0] < width / (2):
        return 1
      else:
        return 0
    else:
      if position[0] > width / 2 - 1:
        return 1
      else:
        return 0

  def ScaredTimer(self, gameState):
    return gameState.getAgentState(self.index).scaredTimer

  def getNextPositions(self, p):
    nextPositions = filter(lambda p: p in self.legalPositions, [(
        p[0] - 1, p[1]), (p[0] + 1, p[1]), (p[0], p[1] - 1), (p[0], p[1] + 1), (p[0], p[1])])
    dist = util.Counter()

    for nextPosition in nextPositions:
      dist[nextPosition] = 1

    return dist

  def getMinDistanceToBorder(self, nextPosition, gameState):
    minDistance = float('inf')

    for borderGrid in self.borderGrids:
      minDistance = min(minDistance, self.getMazeDistance(nextPosition, borderGrid))

    return minDistance

  def getInvaders(self, gameState):
    return [opponent for opponent in self.opponents if gameState.getAgentState(opponent).isPacman]

  def getMinFoodDistance(self, position, foodList):
    minDistance = float('inf')
    targetFood = None

    for food in foodList:
      foodDistance = self.getMazeDistance(position, food)
      if foodDistance < minDistance:
        minDistance = foodDistance
        targetFood = food

    return (minDistance, targetFood)

  # Looks at how an agent could move from where they currently are
  def updateObservations(self, gameState):
    for agent, observation in enumerate(GlobalState.observations):
      if agent in self.opponents:
        newObservation = util.Counter()
        opponentPosition = gameState.getAgentPosition(agent)

        if opponentPosition != None:
          newObservation[opponentPosition] = 1
        else:
          for position in observation:
            if position in self.legalPositions and observation[position] > 0:
              nextPositions = self.getNextPositions(position)

              for x, y in nextPositions:
                newObservation[x, y] += observation[position] * nextPositions[x, y]

          if len(newObservation) == 0:
            oldState = self.getPreviousObservation()

            if oldState != None and oldState.getAgentPosition(agent) != None:  # just ate an enemy
              newObservation[oldState.getInitialAgentPosition(agent)] = 1
            else:
              for p in self.legalPositions:
                newObservation[p] = 1

        GlobalState.observations[agent] = newObservation

  def updateDefendingAgent(self, gameState):
    if GlobalState.defendingAgent != -1:
      return

    invaders = self.getInvaders(gameState)

    if len(invaders) == 0:
      GlobalState.defendingAgent = -1
    else:
      minDistance = float('inf')

      for agentIndex in self.agentsOnTeam:
        invaderDistance = self.getMinInvaderDistanceByAgentIndex(gameState, agentIndex)

        if invaderDistance < minDistance:
          minDistance = invaderDistance
          GlobalState.defendingAgent = agentIndex

  def updateAgentState(self, gameState, nextAction):
    successor = self.getSuccessor(gameState, nextAction)
    nextPosition = successor.getAgentState(self.index).getPosition()

    if nextPosition in self.getFood(gameState).asList():
      self.holdFoodNumber += 1
    if not self.isInEnemyTerritory(successor):
      self.holdFoodNumber = 0

  def updateAvailableFoods(self, gameState, action):
    successor = self.getSuccessor(gameState, action)
    nextPosition = successor.getAgentState(self.index).getPosition()
    targetFood = self.getMinFoodDistance(nextPosition, )[1]

  # Looks for where the enemies currently are
  def observe(self, agent, noisyDistance, currentPosition, gameState):
    allPossible = util.Counter()

    for p in self.legalPositions:
      trueDistance = util.manhattanDistance(currentPosition, p)
      allPossible[p] += gameState.getDistanceProb(trueDistance, noisyDistance)

    for p in self.legalPositions:
      GlobalState.observations[agent][p] *= allPossible[p]

  def determineEvaluateType(self, currentPosition, gameState):
    evaluateType = ATTACK_MODE

    if self.hasReachedBorder == False:
      evaluateType = START_MODE

    if currentPosition == self.entryPosition and self.hasReachedBorder == False:
      self.hasReachedBorder = True
      evaluateType = ATTACK_MODE

    if (GlobalState.defendingAgent == -1 or GlobalState.defendingAgent == self.index) and len(self.getInvaders(gameState)) > 0:
      for opponent in self.opponents:
        if gameState.getAgentState(opponent).isPacman:
          evaluateType = HUNT_MODE
          GlobalState.defendingAgent = self.index
          break

      if not self.isInEnemyTerritory(gameState):
        invaderPositions = self.getInvaderPositions(gameState)

        if len(invaderPositions) > 0:
          for enemy, pos in invaderPositions:
            if self.getMazeDistance(currentPosition, pos) <= 5:
              evaluateType = DEFEND_MODE
              GlobalState.defendingAgent = self.index
              break

    return evaluateType

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    actions.remove('Stop')

    noisyDistances = gameState.getAgentDistances()
    currentPosition = self.getCurrentPosition(gameState)

    # Observe each opponent to get noisy distance measurements
    for agent in self.opponents:
      self.observe(agent, noisyDistances[agent], currentPosition, gameState)

    for opponent in self.opponents:
      GlobalState.observations[opponent].normalize()
      self.potentialPossition[opponent] = max(GlobalState.observations[opponent].items(), key=operator.itemgetter(1))[0]

    self.updateObservations(gameState)
    self.updateDefendingAgent(gameState)

    evaluateType = self.determineEvaluateType(currentPosition, gameState)
    # print('---------------------------------')

    # When it's the first agent round, initialize potentialTargetFoods
    if self.isFirstAgent:
      GlobalState.potentialTargetFoods = set()

    values = [self.evaluate(gameState, action, evaluateType) for action in actions]
    bestActions = [a for a, v in zip(actions, values) if v == max(values)]
    selectedAction = random.choice(bestActions)

    # When it's the second agent round, reset potentialTargetFoods
    if not self.isFirstAgent:
      GlobalState.potentialTargetFoods = set()

    print('selected action:', selectedAction)
    print('---------------------------------')

    self.updateAgentState(gameState, selectedAction)

    return selectedAction

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    position = successor.getAgentState(self.index).getPosition()

    if position != nearestPoint(position):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  # Calculate the heurisic score of each action depending on what tactic is being used
  def evaluate(self, gameState, action, evaluateType):
    """
    Computes a linear combination of features and feature weights
    """
    if evaluateType == ATTACK_MODE:
      print(self.index, "ATTACKING!!!", action)
      features = self.getFeaturesAttack(gameState, action)
      weights = self.getWeightsAttack(gameState, action)
    elif evaluateType == DEFEND_MODE:
      print(self.index, "DEFENDING!!!", action)
      features = self.getFeaturesDefend(gameState, action)
      weights = self.getWeightsDefend(gameState, action)
    elif evaluateType == START_MODE:
      print(self.index, "STARTING!!!", action)
      features = self.getFeaturesStart(gameState, action)
      weights = self.getWeightsStart(gameState, action)
    elif evaluateType == HUNT_MODE:
      print(self.index, "HUNTING!!!", action)
      features = self.getFeaturesHunt(gameState, action)
      weights = self.getWeightHunt(gameState, action)

    print(features)

    return features * weights

  def getFeaturesAttack(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    nextPosition = successor.getAgentState(self.index).getPosition()
    capsules = self.getCapsules(successor)
    ghostDistance = self.getMinGhostDistance(successor)
    scaredGhostDistance = self.getMinScaredGhostDistance(successor)
    willEatFood = False
    isGhostScared = False
    foodList = self.getFood(successor).asList()

    if not self.isFirstAgent and len(foodList) > len(GlobalState.potentialTargetFoods):
      foodList = list(set(foodList) - GlobalState.potentialTargetFoods)

    # Feature successorScore
    features['successorScore'] = -len(foodList)

    # Feature holdFoodNumber
    if nextPosition in self.getFood(gameState).asList():
      features['holdFoodNumber'] = self.holdFoodNumber + 1
      willEatFood = True
    elif not self.isInEnemyTerritory(successor):
      features['holdFoodNumber'] = 0
    else:
      features['holdFoodNumber'] = self.holdFoodNumber

    # Feature foodDistance
    if len(foodList) > 0:
      minDistance, targetFood = self.getMinFoodDistance(nextPosition, foodList)
      features['foodDistance'] = minDistance

      # If it's the first agent, update potentialTargetFoods
      if self.isFirstAgent:
        GlobalState.potentialTargetFoods.add(targetFood)

    if willEatFood:
      features['foodDistance'] *= 0.5
    else:
      if self.holdFoodNumber <= 2:
        features['foodDistance'] *= 5

    # Feature eatCapsule and capsuleDistance
    if len(capsules) > 0:
      minCapsuleDistance = min([self.getMazeDistance(nextPosition, capsule) for capsule in capsules])
    else:
      minCapsuleDistance = 0

    features['eatCapsule'] = -len(capsules)
    features['capsuleDistance'] = minCapsuleDistance

    # Feature ghostDistance
    features['ghostDistance'] = ghostDistance if ghostDistance != -1 else 100000

    if features['ghostDistance'] >= 8:
      features['ghostDistance'] = 100000

      if not willEatFood:
        features['foodDistance'] *= 4
    elif features['ghostDistance'] >= 5:
      features['ghostDistance'] *= 2

    # Feature scaredGhostDistance
    features['scaredGhostDistance'] = scaredGhostDistance if scaredGhostDistance != -1 else 0

    if scaredGhostDistance != -1 and ghostDistance == scaredGhostDistance:
      features['ghostDistance'] = 100000
      isGhostScared = True

    # Feature homeDistance
    if self.isInEnemyTerritory(successor):
      minDistanceToBorder = self.getMinDistanceToBorder(nextPosition, successor)
      features['homeDistance'] = minDistanceToBorder

      if minCapsuleDistance <= minDistanceToBorder:
        features['capsuleDistance'] *= 2
    else:
      features['homeDistance'] = 0

    if self.holdFoodNumber > 5 and not features['ghostDistance'] >= 8:
      features['homeDistance'] *= 100
    elif self.holdFoodNumber > 3:
      features['homeDistance'] *= 2

    if ghostDistance > 0 and ghostDistance <= 2 and not isGhostScared:
      features['homeDistance'] *= 5

    # Feature partnerDistance
    features['partnerDistance'] = self.getPartnerDistance(successor)

    if features['partnerDistance'] <= 3:
      features['partnerDistance'] *= 5

    # Feature deadEnd
    features['deadEnd'] = GlobalState.getDangerLevel(nextPosition)

    if isGhostScared:
      features['deadEnd'] = 0
    else:
      if ghostDistance > 0 and ghostDistance <= 5:
        features['deadEnd'] *= 1000
        features['homeDistance'] *= 5

      if not self.isInEnemyTerritory(successor):
        features['deadEnd'] *= 0.1

    # Feature stop
    if action == Directions.STOP:
      features['stop'] = 1
      features['deadEnd'] *= 1000

      if not self.isInEnemyTerritory(successor):
        features['stop'] *= 100
    else:
      features['stop'] = 0

    return features

  def getFeaturesHunt(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    nextPosition = successor.getAgentState(self.index).getPosition()
    isInEnemyTerritory = self.isInEnemyTerritory(successor)
    ghostDistance = self.getMinGhostDistance(successor) if isInEnemyTerritory else 10000

    invaders = [opponent for opponent in self.opponents if successor.getAgentState(opponent).isPacman]
    features['invadersNumber'] = len(invaders)

    # Feature invaderDistance
    features['invaderDistance'] = float('inf')

    for invader in invaders:
      enemyPosition = self.potentialPossition[invader]
      enemyDistance = self.getMazeDistance(nextPosition, enemyPosition)
      features['invaderDistance'] = min(features['invaderDistance'], enemyDistance)

    # Feature homeDistance
    if isInEnemyTerritory:
      minDistanceToBorder = self.getMinDistanceToBorder(nextPosition, successor)
      features['homeDistance'] = minDistanceToBorder
    else:
      features['homeDistance'] = 0

    # Feature ghostDistance
    features['ghostDistance'] = ghostDistance

    # Feature deadEnd
    features['deadEnd'] = GlobalState.getDangerLevel(nextPosition) if self.isInEnemyTerritory(successor) else 0

    if action == Directions.STOP:
      features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev:
      features['reverse'] = 1

    return features

  def getFeaturesDefend(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    nextPosition = successor.getAgentState(self.index).getPosition()

    invaders = self.getInvaders(successor)
    features['invadersNumber'] = len(invaders)

    if len(invaders) < len(self.getInvaders(gameState)):
      GlobalState.defendingAgent = -1

    invaderDistance = self.getMinInvaderDistance(successor)

    if len(invaders) > 0:
      features['invaderDistance'] = invaderDistance

    if invaderDistance <= 5:
      features['danger'] = 1
      if invaderDistance <= 1 and self.ScaredTimer(successor) > 0:
        features['danger'] = -1
    else:
      features['danger'] = 0

    if action == Directions.STOP:
      features['stop'] = 1

    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev:
      features['reverse'] = 1

    return features

  # Returns all the heuristic features for the START tactic
  def getFeaturesStart(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    nextPosition = successor.getAgentState(self.index).getPosition()

    features['distToEntryPosition'] = self.getMazeDistance(nextPosition, self.entryPosition)

    if nextPosition == self.entryPosition:
      features['hasReachedBorder'] = 1

    return features

  def getWeightsAttack(self, gameState, action):
    return {'successorScore': 50000, 'foodDistance': -3000, 'holdFoodNumber': -1000,
            'eatCapsule': 10000000, 'capsuleDistance': -500, 'ghostDistance': 25000, 'scaredGhostDistance': -10000,
            'homeDistance': -2000, 'partnerDistance': 3000, 'deadEnd': 15, 'stop': -50000}

  def getWeightHunt(self, gameState, action):
    return {'invadersNumber': -1000, 'invaderDistance': -100, 'stop': -5000,
            'reverse': 0, 'homeDistance': -5000, 'ghostDistance': 5000, 'deadEnd': 200}

  def getWeightsDefend(self, gameState, action):
    return {'invadersNumber': -30000, 'invaderDistance': -500, 'stop': -5000,
            'reverse': -200, 'danger': 3000}

  def getWeightsStart(self, gameState, action):
    return {'distToEntryPosition': -1, 'hasReachedBorder': 1000}

  def setEntryPosition(self, gameState):
    positions = []
    centerX = round(gameState.getWalls().width / 2)
    centerY = round(gameState.getWalls().height / 2)
    maxHeight = gameState.getWalls().height - centerY if self.isTopAgent else centerY
    self.entryPosition = (centerX, centerY)

    for i in range(maxHeight):
      if not gameState.hasWall(centerX, centerY):
        positions.append((centerX, centerY))

      centerY = centerY + 1 if self.isTopAgent else centerY - 1

    currentPosition = self.getCurrentPosition(gameState)
    minDist = float('inf')

    # Find shortest distance to the entry position
    for position in positions:
      dist = self.getMazeDistance(currentPosition, position)

      if dist <= minDist:
        minDist = dist
        self.entryPosition = position


class TopAgent(MainAgent):

  def start(self, gameState):
    self.setEntryPosition(gameState)


class BottomAgent(MainAgent):

  def start(self, gameState):
    self.setEntryPosition(gameState)
