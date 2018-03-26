import numpy as np
import collections
import random

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers import Dropout

from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D
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
    def __init__(self,numcases=16,gridsize=(4,4)):
        """Init a new agent.
        """
        self.numcases=numcases
        self.gridsize=gridsize

        self.q = collections.defaultdict(lambda: np.zeros((self.numcases,), dtype=np.float32))
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

        s = observation
        # choose action
        eps = 1.0/np.sqrt(self.t)
        if np.random.rand(0,1) > eps:
            pos_index=np.random.randint(0,self.numcases)
            pos = (pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            return pos
        else:
            pos_index= np.argmax(self.q[s])
            pos=(pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            return pos


    def reward(self, obs, act, reward):
        self.pending = (obs, act, reward)


####################################################CONVO NETWORK        



class QAgentConvoNetwork:
    def __init__(self,gridsize=(4,4)):
        """Init a new agent.
        """
        self.numcases=gridsize[0]*gridsize[1]
        self.gridsize=gridsize

        self.MemorySize=1000 #Each time the container ExperienceReplay is full we fit the new data to our Conv neural net with SampleToFit observation
        self.SampleToFit=int(self.MemorySize*0.7) #70% of the data stored in ExperienceReplay is used for fitting the conv
        self.ExperienceReplay=[]
        self.batch_size=50
        self.epochs=10
        self.forgetRate=3.0/2 #the amount of infromation we choose to forget in the ExperienceReplay container

        self.gamma = 0.9
        self.t = 1
        
        self.q = collections.defaultdict(lambda: np.zeros((self.numcases,), dtype=np.float32))


        self.pending = None
        self.newgame = True

        # Customize your network here
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(gridsize[0],gridsize[1],1)))
        self.model.add(Flatten())
        self.model.add(Dense(gridsize[0]*gridsize[1],activation='relu'))

        ##compile model
        sgd = SGD(lr=0.01,
                decay=1e-6, momentum=0.9,
                nesterov=True)

        self.model.compile(loss='mean_squared_error',
                        optimizer=sgd)
    def reset(self):
        """Reset the internal state of the agent, for a new run.

        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.

        You must **not** reset the learned parameters.
        """
        self.newgame = True

    def to_pixels(self,tt):
        image1=np.zeros(self.gridsize)
        
        for i in range(self.numcases):
            i_pos = (i // self.gridsize[1], i % self.gridsize[1])

            if tt[i]=="X":
                image1[i_pos[0],i_pos[1]]=255/2
            elif tt[i]=="M":
                image1[i_pos[0],i_pos[1]]=255
            else:
                image1[i_pos[0],i_pos[1]]=float(tt[i])*10
        return image1

    def processTrainData(self,train):
        output=[]
        for tt in train:
            image1=self.to_pixels(tt[0])
            image2=self.to_pixels(tt[1])
            output.append(image2-image1)
        return np.array(output)
            
    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
        """

        if self.t%self.MemorySize==0:
            #Train the model
            mybatch=random.sample(self.ExperienceReplay,self.SampleToFit)#[:self.SampleToFit]
            train_not_processed=[(myb[0],myb[1]) for myb in mybatch]
            train=self.processTrainData(train_not_processed)
            train=train.reshape(train.shape[0],self.gridsize[0],self.gridsize[1],1)
            targets=np.array([vec[2].reshape(16,) for vec in mybatch])
            print("####-------> fitting the convolutional network with new data....")
            self.model.fit(train,targets,epochs=self.epochs, batch_size=self.batch_size)

            #remove the part we wish to forget in the experience replay
            start=int(self.MemorySize*self.forgetRate)
            self.ExperienceReplay=self.ExperienceReplay[start:]


        if self.pending is not None:
            #if self.newgame:
                #s = "FINISH"
                #self.newgame = False
            #else:
            s = observation
            self.t += 1
            (last_s, last_a, last_r) = self.pending
            pos_index = 0
            pos_index_temp = range(last_a[0]*self.gridsize[1], (last_a[0]+1)*self.gridsize[1])
            for i in pos_index_temp:
                if i%self.gridsize[1] == last_a[1]:
                    pos_index = i
                    break


            qsa = self.q[last_s][pos_index]
            target_qsa = last_r + self.gamma * self.q[s].max()
            target_qsa_vec=self.q[last_s]
            target_qsa_vec[pos_index]=target_qsa
            
            to_add=(last_s.split(' '),s.split(' '),target_qsa_vec)
            
            self.ExperienceReplay.append(to_add)

            #use the convo network to compute the update of the q-value
            #process to_add before prediction
            if self.t>self.MemorySize:
                to_add_not_processed=[(myb[0],myb[1]) for myb in [to_add]]
                to_add=self.processTrainData(to_add_not_processed)
                to_add=to_add.reshape(to_add.shape[0],self.gridsize[0],self.gridsize[1],1)
                self.q[last_s] = self.model.predict(to_add)

            self.pending = None 

        s = observation
        # choose action
        eps = 1.0/np.sqrt(self.t)
        if np.random.rand(0,1) > eps and self.t>self.MemorySize:
            pos_index=np.random.randint(0,self.numcases)
            pos = (pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            return pos
        else:
            pos_index= np.argmax(self.q[s])
            pos=(pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            return pos


    def reward(self, obs, act, reward):
        self.pending = (obs, act, reward)
#Agent = QAgent
#Agent =RandomAgent
Agent=QAgentConvoNetwork
