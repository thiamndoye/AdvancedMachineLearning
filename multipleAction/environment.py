from __future__ import division
from __future__ import print_function
import numpy as np


"""
This file contains the definition of the environment
in which the agents are run.
"""
#Possible actions are:
#click
#flag
#unfalg

class Environment:
    # List of the possible actions by the agents
    possible_actions = []

    def __init__(self, gridsize=(3, 3), num_mines=2):
        """Instanciate a new environement without placing the mines.
        """
        self.gridsize = gridsize
        self.num_cases = gridsize[0]*gridsize[1]
        self.num_mines = num_mines
        self.discovered_cases = 0
        self.gameFinishedCount=0
        self.reset()

        self.flagged=0


    def reset(self):
        """Reset the environment for a new run."""
        self.discovered_cases = 0
        self.flagged=0
        self.mines_pos = []
        self.screen = {}

        for i in range(self.num_cases):
            i_pos = (i // self.gridsize[1], i % self.gridsize[1])
            self.screen[i_pos] = "X"
        self.positions = list(self.screen.keys())

    def observe(self):
        """Returns the current observation that the agent can make
        of the environment, if applicable.
        """
        observation = self.screen.copy()
        state=[]
        for i in range(self.num_cases):
            i_pos = (i // self.gridsize[1], i % self.gridsize[1])
            state.append(str(observation[i_pos]))

        return (" ".join(state))

    def mines_placing(self, action):
        """After first action, the mines are placed in every other position than the one form the first action.
        """
        idx_action = 0
        idx_temp_act = range(action[0]*self.gridsize[1], (action[0]+1)*self.gridsize[1])
        for i in idx_temp_act:
            if i%self.gridsize[1] == action[1]:
                idx_action = i
                break
        available_indx = [x for x in range(self.num_cases) if x != idx_action]
        mines_idx = np.random.choice(available_indx, self.num_mines, replace=False)
        self.mines_pos = [(n//self.gridsize[1], n%self.gridsize[1]) for n in mines_idx]

    def grid_valuation(self):
        """ Create a dictionary with the real values of each case """
        self.value = {}
        for i in range(self.num_cases):
            i_pos = (i//self.gridsize[1], i%self.gridsize[1])
            if i_pos in self.mines_pos:
                self.value[i_pos] = "M"
            else:
                danger = 0
                for x in [i_pos[0]-1, i_pos[0], i_pos[0]+1]:
                    for y in [i_pos[1]-1, i_pos[1], i_pos[1]+1]:
                        if (x,y) in self.mines_pos:
                            danger += 1
                self.value[i_pos] = danger

    def adjacent(self, position):
        """ Returns the 8 adjacent positions (if present inside the field)"""
        adjacent_list = []
        for x in [position[0]-1, position[0], position[0]+1]:
            for y in [position[1]-1, position[1], position[1]+1]:
                if (x,y)!=position and (x,y) in self.positions :
                    adjacent_list.append((x,y))

        return adjacent_list

    def discover(self, box):
        """ Discover the values of the 8 adjacent boxes adjacent to one box discovered with value = 0"""
        self.screen[box] = self.value[box]
        adjacent_positions = self.adjacent(box)
        for position in adjacent_positions:
            if self.value[position] == 0 and self.screen[position] == "X":
                self.discover(position)
            self.screen[position] = self.value[position]


    def act(self, action):
        """Perform given action by the agent on the environment,
        and returns a reward.
        """
        if self.discovered_cases == 0:
            if len(action)!=2:
                action=(action[0],action[1])

            self.mines_placing(action)
            self.grid_valuation()
            self.discovered_cases+=1
            return (0,None)
        prev_discovered = self.discovered_cases

        if len(action)==2:
            if action in self.mines_pos:
                return (-10.0, "Boom")
            elif self.screen[action] != "X":
                return(-1.0, None)
            else:
                self.screen[action] = self.value[action]
                if self.value[action] == 0:
                    self.discover(action)
                self.discovered_cases = self.num_cases - sum(1 for x in self.screen.values() if (x == "X" or x=="F"))
                score = self.discovered_cases - prev_discovered
                if self.discovered_cases == self.num_cases - self.num_mines:
                    event = "End game"
                    self.gameFinishedCount+=1
                else:
                    event = None
                return(score, event)
        else:
            pos=(action[0],action[1])
            if action[2]=="F":
                self.flagged+=1
                self.screen[pos]="F"
                if self.flagged>self.num_mines:
                    return(-2.0,None)
                elif pos in self.mines_pos :
                    return(2.0,None)
                else:
                    return(0.0,None)
            elif action[2]=="UF":
                if self.screen[pos]=="F":
                    self.screen[pos]="X"
                    return(0.0,None)
                else:
                    return (-1.0,None)


    def display(self):
        """ Screening the grid for the verbose"""
        print("+-", end='')
        for x in range(self.gridsize[1]):
            print("--", end='')
        print("+ ")
        for y in range(self.gridsize[0]-1, -1, -1):
            print("| ", end='')
            for x in range(self.gridsize[1]):
                pos = (y, x)
                print(self.screen[pos], end='')
                print(" ", end='')
            print("|")
        print("+-", end='')
        for x in range(self.gridsize[1]):
            print("--", end='')
        print("+")
