# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor) 
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if action == Directions.STOP:
            return -999999

        # manhattan distance to nearest food in successor state
        minDistToFood = 999999
        for foodPos in newFood.asList():
            minDistToFood = min(manhattanDistance(foodPos, newPos), minDistToFood)

        minDistToCaps = 999999
        for capsulePos in newCapsules:
            minDistToCaps = min(manhattanDistance(capsulePos, newPos), minDistToCaps)

        minDistToGhost = 999999
        if currentCapsules == newCapsules:
            for ghost in newGhostStates:
                ghostPos = ghost.getPosition()
                if newPos == ghostPos:
                    return successorGameState.getScore() - 999999
                minDistToGhost = min(manhattanDistance(ghostPos, newPos), minDistToGhost)

        return successorGameState.getScore()  + minDistToGhost/minDistToFood + 1/minDistToCaps

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # find action that leads to max value using eval funct by checking down to given depth limit
        # only care about first action, rest just looking at game states

        maxi = -999999 # initialize max score
        best_action = Directions.STOP # initialize best action to Stop

        # for Pacman's legal actions, find action with greatest estimated score using minimax alg
        for action in gameState.getLegalActions(0):

            nextState = gameState.generateSuccessor(0, action)
            tmp = self.min_value(nextState, 1, self.depth)

            if tmp > maxi: # better action found
                maxi = tmp
                best_action = action

        return best_action


    def max_value(self, gameState, agent, depth):
        if depth <= 0 or gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(agent)) == 0:
            return self.evaluationFunction(gameState)

        val = -999999
        for action in gameState.getLegalActions(agent):
            val = max(val, self.min_value(gameState.generateSuccessor(agent, action), agent + 1, depth))
        return val

    def min_value(self, gameState, agent, depth):
        if depth <= 0 or gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(agent)) == 0:
            return self.evaluationFunction(gameState)

        val = 999999
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)

            if agent == gameState.getNumAgents() - 1: # no more ghost agents, next ply
                val = min(val, self.max_value(successor, 0, depth-1))
            else:
                val = min(val, self.min_value(successor, agent + 1, depth))
        return val

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxi = -999999
        best_action = None

        alpha = -999999
        beta = 999999
        for action in gameState.getLegalActions(0):

            nextState = gameState.generateSuccessor(0, action)
            tmp = self.min_value(nextState, 1, self.depth, alpha, beta)

            if tmp > maxi:
                maxi = tmp
                best_action = action

            alpha = max(alpha, tmp)

        return best_action

    def max_value(self, gameState, agent, depth, alpha, beta):
        # alpha : MAX's best option, beta : MIN's best option
        if depth <= 0 or gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(agent)) == 0:
            return self.evaluationFunction(gameState)

        val = -999999
        for action in gameState.getLegalActions(agent):
            val = max(val, self.min_value(gameState.generateSuccessor(agent, action), agent + 1, depth, alpha, beta))
            if val > beta:
                return val
            alpha = max(alpha, val)
        return val

    def min_value(self, gameState, agent, depth, alpha, beta):
        # alpha : MAX's best option, beta : MIN's best option
        if depth <= 0 or gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(agent)) == 0:
            return self.evaluationFunction(gameState)

        val = 999999
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)

            if agent == gameState.getNumAgents() - 1:
                val = min(val, self.max_value(successor, 0, depth - 1, alpha, beta))
            else:
                val = min(val, self.min_value(successor, agent + 1, depth, alpha, beta))
            if val < alpha:
                return val
            beta = min(beta, val)

        return val


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        maxi = -999999
        best_action = Directions.STOP

        for action in gameState.getLegalActions(0):

            nextState = gameState.generateSuccessor(0, action)
            tmp = self.exp_value(nextState, 1, self.depth)

            if tmp > maxi:
                maxi = tmp
                best_action = action

        return best_action

    def max_value(self, gameState, agent, depth):
        if depth <= 0 or gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(agent)) == 0:
            return self.evaluationFunction(gameState)

        val = -999999
        for action in gameState.getLegalActions(agent):
            val = max(val, self.exp_value(gameState.generateSuccessor(agent, action), agent + 1, depth))
        return val

    def exp_value(self, gameState, agent, depth):
        if depth <= 0 or gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(agent)) == 0:
            return self.evaluationFunction(gameState)

        val = 0
        prob = 1 / float(len(gameState.getLegalActions(agent)))
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)

            if agent == gameState.getNumAgents() - 1:
                val += prob * self.max_value(successor, 0, depth - 1)
            else:
                val += prob * self.exp_value(successor, agent + 1, depth)
        return val


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # evaluation score = a * 1/(# food) + b * 1/(min manhattan distance to capsules) + (eval for nearest ghost)

    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood().asList()
    currCaps = currentGameState.getCapsules()
    currGhostStates = currentGameState.getGhostStates()

    evalScore = 0
    # coefficients of linear combination
    a, b, c, d = 10, 10, 4, 1

    # to prevent pacman staying still, "encourages" to eat next pellet
    if len(currFood) > 0:
        evalScore += a/len(currFood)

    # go nearer to closest pellet
    if len(currCaps) > 0:
        minDistToCaps = 999999
        for capsulePos in currCaps:
            minDistToCaps = min(manhattanDistance(capsulePos, currPos), minDistToCaps)
        evalScore += b/minDistToCaps

    # evaluation for ghosts: find closest ghost based on manhattan distance
    isScared = False
    minDistToGhost = 999999
    for ghost in currGhostStates:
        ghostPos = ghost.getPosition()
        if currPos == ghostPos:
            return -999999
        minDistToGhost = min(manhattanDistance(ghostPos, currPos), minDistToGhost)
        if ghost.scaredTimer > 0:
            isScared = True

    # if ghosts scared (scared timer > 0), pacman should go closer and eat them!
    # else stay away but don't be paranoid (d=1)
    if isScared:
        evalScore += c/minDistToGhost
    else:
        evalScore += d*minDistToGhost

    # combine evaluated score plus current game state's score
    return evalScore + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

