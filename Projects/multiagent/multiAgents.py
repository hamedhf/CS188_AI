# multiAgents.py
# --------------
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

from pacman import GameState
from util import manhattanDistance
from game import Directions
import random
import util
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # successorGameState:
        # %%%%%%%%%%%%%%%%%%%%
        # %o...%........%....%
        # %.%%.%.%%%%%%.%.%%.%
        # %.%..............%.%
        # %.%.%%.%%  %%.%%.%.%
        # %......%    %......%
        # % %.%%.%%%%%%.%%.%G%
        # % %.   ..........%.%
        # %G%% % %%%%%%.%.%%.%
        # % <  %    ....%...o%
        # %%%%%%%%%%%%%%%%%%%%
        # newPos: (2, 1)
        # newFood:
        # FFFFFFFFFFFFFFFFFFFF
        # FFTTTFTTTTTTTTFTTTTF
        # FTFFTFTFFFFFFTFTFFTF
        # FTFTTTTTTTTTTTTTTFTF
        # FTFTFFTFFFFFFTFFTFTF
        # FTTTTTTFFFFFFTTTTTTF
        # FTFTFFTFFFFFFTFFTFTF
        # FTFTTTTTTTTTTTTTTFTF
        # FTFFTFTFFFFFFTFTFFTF
        # FTTTTFTTTFTTTTFTTTFF
        # FFFFFFFFFFFFFFFFFFFF
        # newGhostStates: [<game.AgentState object at 0x7f8e9d754ac8>, <game.AgentState object at 0x7f8e9d754e10>]
        # newScaredTimes: [30, 30]
        # successorGameState.getScore()

        "*** YOUR CODE HERE ***"
        # food_positions = successorGameState.getFood().asList()
        # ghost_positions = successorGameState.getGhostPositions()
        successor_score = 0

        # walls = currentGameState.getWalls()
        # pos = list(currentGameState.getPacmanPosition())
        # if action == Directions.NORTH:
        #     pos[1] += 1
        # elif action == Directions.SOUTH:
        #     pos[1] -= 1
        # elif action == Directions.EAST:
        #     pos[0] += 1
        # elif action == Directions.WEST:
        #     pos[0] -= 1
        # else:
        #     # stop action
        #     pass

        # # avoid walls
        # if pos in walls:
        #     successor_score -= 10

        # eat ghost if possible, else run
        # for ghost in newGhostStates:
        #     distance_from_ghost = manhattanDistance(
        #         ghost.getPosition(), newPos)

        #     if ghost.scaredTimer > distance_from_ghost:
        #         successor_score += 10
        #     else:
        #         successor_score -= 20

        # ghost_count = len(newGhostStates)
        # foods = currentGameState.getFood()
        # if foods[newPos[0]][newPos[1]]:
        #     successor_score += 1

        # capsules = currentGameState.getCapsules()
        # if newPos in capsules:
        #     successor_score += 10

        # food_positions = foods.asList()
        # nearest_food_distance = manhattanDistance(newPos - food_positions[0])
        # for food_pos in food_positions:
        #     tmp = manhattanDistance(newPos - food_pos)
        #     nearest_food_distance = tmp if tmp < nearest_food_distance else nearest_food_distance

        # successor_score += (-(0.5) * nearest_food_distance)
        return successor_score


def scoreEvaluationFunction(currentGameState: GameState) -> float:
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        legal_moves = gameState.getLegalActions(self.index)
        successor_states = list(
            map(lambda action: gameState.generateSuccessor(self.index, action), legal_moves))
        successor_values = list(
            map(lambda state: self.value(1, state), successor_states))
        idx = successor_values.index(max(successor_values))

        return legal_moves[idx]

    def value(self, agent_index: int, state: GameState):
        if state.isWin() or state.isLose() or agent_index == state.getNumAgents() * self.depth:
            # it’s a terminal state
            return self.evaluationFunction(state)
        else:
            # non terminal state
            return self.max_value(agent_index, state) if agent_index % state.getNumAgents() == 0 else self.min_value(agent_index, state)

    def max_value(self, agent_index: int, state: GameState):
        v = float("-inf")
        legal_moves = state.getLegalActions(agent_index % state.getNumAgents())
        successor_states = list(
            map(lambda action: state.generateSuccessor(agent_index % state.getNumAgents(), action), legal_moves))
        for successor_state in successor_states:
            v = max(v, self.value(agent_index + 1, successor_state))
        return v

    def min_value(self, agent_index: int, state: GameState):
        v = float("inf")
        legal_moves = state.getLegalActions(agent_index % state.getNumAgents())
        successor_states = list(
            map(lambda action: state.generateSuccessor(agent_index % state.getNumAgents(), action), legal_moves))
        for successor_state in successor_states:
            v = min(v, self.value(agent_index + 1, successor_state))
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        legal_moves = gameState.getLegalActions(self.index)
        alpha = float("-inf")
        beta = float("inf")
        successor_states = list(
            map(lambda action: gameState.generateSuccessor(self.index, action), legal_moves))
        successor_values = []
        for state in successor_states:
            v = self.value(1, state, alpha, beta)
            successor_values.append(v)
            alpha = max(v, alpha)
        idx = successor_values.index(max(successor_values))

        return legal_moves[idx]

    def value(self, agent_index: int, state: GameState, alpha: float, beta: float):
        if state.isWin() or state.isLose() or agent_index == state.getNumAgents() * self.depth:
            # it’s a terminal state
            return self.evaluationFunction(state)
        else:
            # non terminal state
            return self.max_value(agent_index, state, alpha, beta) if agent_index % state.getNumAgents() == 0 else self.min_value(agent_index, state, alpha, beta)

    def max_value(self, agent_index: int, state: GameState, alpha: float, beta: float):
        v = float("-inf")
        legal_moves = state.getLegalActions(agent_index % state.getNumAgents())
        for i in range(len(legal_moves)):
            v = max(v, self.value(agent_index + 1, state.generateSuccessor(agent_index %
                    state.getNumAgents(), legal_moves[i]), alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, agent_index: int, state: GameState, alpha: float, beta: float):
        v = float("inf")
        legal_moves = state.getLegalActions(agent_index % state.getNumAgents())
        for i in range(len(legal_moves)):
            v = min(v, self.value(agent_index + 1, state.generateSuccessor(agent_index %
                    state.getNumAgents(), legal_moves[i]), alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v


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
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
