import numpy as np
import collections

"""
Contains the definition of the agent that will run in an
environment.
"""


class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.q = collections.defaultdict(lambda: np.zeros((3,), dtype=np.float32))
        self.gamma = 0.9
        self.t = 1
        self.pending = None
        self.newgame = True

    def reset(self):
        """Reset the internal state of the agent, for a new run.

        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.

        You must **not** reset the learned parameters.
        """
        pass

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
        """
        x = np.random.randint(0, 3)
        y = np.random.randint(0, 3)
        return((x,y))

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass

class QAgent:
    def __init__(self,numcases=9,gridsize=(3,3)):
        """Init a new agent.
        """
        self.numcases=numcases
        self.gridsize=gridsize

        self.q = collections.defaultdict(lambda: np.zeros((self.numcases*3,), dtype=np.float32))
        self.gamma = 0.9
        self.t = 1
        

        self.pending = None
        self.newgame = True

    def reset(self):
        """Reset the internal state of the agent, for a new run.

        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.

        You must **not** reset the learned parameters.
        """
        self.newgame = True

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
        """
        if self.pending is not None:
            if self.newgame:
                s = "FINISH"
                self.newgame = False
            else:
                s = observation
            self.t += 1
            (last_s, last_a, last_r) = self.pending
            if len(last_a)==2:
                pos_index = 0
                pos_index_temp = range(last_a[0]*self.gridsize[1], (last_a[0]+1)*self.gridsize[1])
                for i in pos_index_temp:
                    if i%self.gridsize[1] == last_a[1]:
                        pos_index = i
                        break

                qsa = self.q[last_s][pos_index]
                alpha = 1.0/self.t
                new_q = qsa + alpha * ( last_r + self.gamma * self.q[s].max() - qsa )
                self.q[last_s][pos_index] = new_q
                self.pending = None
            else:
                if last_a[2]=="F":
                    pos_index = 0
                    pos_index_temp = range(last_a[0]*self.gridsize[1], (last_a[0]+1)*self.gridsize[1])
                    for i in pos_index_temp:
                        if i%self.gridsize[1] == last_a[1]:
                            pos_index = i
                            break
                    pos_index=pos_index+9
                    qsa = self.q[last_s][pos_index]
                    alpha = 1.0/self.t
                    new_q = qsa + alpha * ( last_r + self.gamma * self.q[s].max() - qsa )
                    self.q[last_s][pos_index] = new_q
                    self.pending = None
                elif last_a[2]=="UF":
                    pos_index = 0
                    pos_index_temp = range(last_a[0]*self.gridsize[1], (last_a[0]+1)*self.gridsize[1])
                    for i in pos_index_temp:
                        if i%self.gridsize[1] == last_a[1]:
                            pos_index = i
                            break
                    pos_index=pos_index+9
                    qsa = self.q[last_s][pos_index]
                    alpha = 1.0/self.t
                    new_q = qsa + alpha * ( last_r + self.gamma * self.q[s].max() - qsa )
                    self.q[last_s][pos_index] = new_q
                    self.pending = None

        s = observation
        # choose action
        eps = 1.0/np.sqrt(self.t)
        if np.random.rand(0,1) > eps:
            pos_index=np.random.randint(0,self.numcases)
            pos = (pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            act=np.random.randint(0,3)
            if act==0:
                return pos
            elif act==1:
                return (pos[0],pos[1],"F")
            else:
                return (pos[0],pos[1],"UF")

        else:
            pos_index= np.argmax(self.q[s])
            if pos_index<9:
                pos=(pos_index // self.gridsize[1], pos_index % self.gridsize[1])
                return pos
            elif pos_index>=9 and pos_index<18:
                pos=((pos_index-9) // self.gridsize[1], (pos_index-9) % self.gridsize[1])
                return (pos[0],pos[1],"F")
            else:
                pos=((pos_index-18) // self.gridsize[1], (pos_index-18) % self.gridsize[1])
                return (pos[0],pos[1],"UF")


    def reward(self, obs, act, reward):
        

        self.pending = (obs, act, reward)


Agent = QAgent#(9,(3,3))
