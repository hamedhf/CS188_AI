# search.py
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from platform import node
from typing import Union
import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


class Node:
    def __init__(self, state, old_actions: list, new_action: str) -> None:
        self.state = state
        self.actions = old_actions
        if new_action != "":
            self.actions.append(new_action)


class CostedNode(Node):
    def __init__(self, state, old_actions: list, new_action: str, cost: Union[int, float]) -> None:
        super().__init__(state, old_actions, new_action)
        self.cost = cost


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    Start: (5, 5)

    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    Is the start a goal? False

    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]

    """
    "*** YOUR CODE HERE ***"

    fringe: util.Stack = util.Stack()
    closed_set: list = []
    start_node: Node = Node(problem.getStartState(), [], "")
    fringe.push(start_node)

    while not fringe.isEmpty():
        current_node: Node = fringe.pop()

        if problem.isGoalState(current_node.state):
            return current_node.actions
        else:
            # check if it’s already expanded or not
            if current_node.state in closed_set:
                continue

            successors = problem.getSuccessors(current_node.state)
            closed_set.append(current_node.state)

            for successor in successors:
                successor_node: Node = Node(
                    successor[0], current_node.actions.copy(), successor[1])
                fringe.push(successor_node)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    fringe: util.Queue = util.Queue()
    closed_set: list = []
    start_node: Node = Node(problem.getStartState(), [], "")
    fringe.push(start_node)

    while not fringe.isEmpty():
        current_node: Node = fringe.pop()

        if problem.isGoalState(current_node.state):
            return current_node.actions
        else:
            # check if it’s already expanded or not
            if current_node.state in closed_set:
                continue

            successors = problem.getSuccessors(current_node.state)
            closed_set.append(current_node.state)

            for successor in successors:
                successor_node: Node = Node(
                    successor[0], current_node.actions.copy(), successor[1])
                fringe.push(successor_node)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    fringe = util.PriorityQueue()
    closed_set: list = []
    start_node: CostedNode = CostedNode(problem.getStartState(), [], "", 0)
    fringe.push(start_node, 0)

    while not fringe.isEmpty():
        current_node: CostedNode = fringe.pop()

        if problem.isGoalState(current_node.state):
            return current_node.actions
        else:
            # check if it’s already expanded or not
            if current_node.state in closed_set:
                continue

            successors = problem.getSuccessors(current_node.state)
            closed_set.append(current_node.state)

            for successor in successors:
                successor_node: CostedNode = CostedNode(successor[0], current_node.actions.copy(
                ), successor[1], current_node.cost + successor[2])
                fringe.push(successor_node, successor_node.cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    fringe = util.PriorityQueue()
    closed_set: list = []
    start_node: CostedNode = CostedNode(problem.getStartState(
    ), [], "", 0)
    fringe.push(start_node, start_node.cost +
                heuristic(start_node.state, problem))

    while not fringe.isEmpty():
        current_node: CostedNode = fringe.pop()

        if problem.isGoalState(current_node.state):
            return current_node.actions
        else:
            # check if it’s already expanded or not
            if current_node.state in closed_set:
                continue

            successors = problem.getSuccessors(current_node.state)
            closed_set.append(current_node.state)

            for successor in successors:
                successor_node: CostedNode = CostedNode(successor[0], current_node.actions.copy(
                ), successor[1], current_node.cost + successor[2])
                fringe.push(successor_node, successor_node.cost +
                            heuristic(successor_node.state, problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
