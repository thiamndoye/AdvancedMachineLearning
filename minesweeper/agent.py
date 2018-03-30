import numpy as np
import collections
import random
import os
import json

from os.path import join
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers import Dropout

from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.models import model_from_json
"""
Contains the definition of the agent that will run in an
environment.
"""
#Save a model to disk
def saveModel(model,name,q):
    # serialize model to JSON
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+"_weights.h5")
    with open(name+"_q_value_records.json", 'w') as f:
        dict_to_save={a:q[a].tolist() for a in q}
        json.dump(dict_to_save, f)

    print("Saved model to disk")
#Load a model from disk
def loadModel(name):
    # load json and create model
    json_file = open(name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    sgd = SGD(lr=0.01,
                decay=1e-6, momentum=0.9,
                nesterov=True)

    loaded_model.compile(loss='mean_squared_error',
                        optimizer=sgd)
    print("Model compiled correctly")
    loaded_model.load_weights(name+"_weights.h5")
    print("Loaded model from disk. Please compile it before usage")
    return(loaded_model)

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

class BanditAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.x_max = 0
        self.y_max = 0
        self.turn = 0
        self.game = 0
        self.sub_squares = []
        self.dict_obs = {}
        self.learn_dict = {}
        self.epsilon = 0.2
        self.sub_state_keys = []

    def reset(self):
        """Reset the internal state of the agent, for a new run.

        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.

        You must **not** reset the learned parameters.
        """
        self.turn = 0
        self.game += 1
        pass

    def load_last_learn_dict(self):
        list_of_files = glob.glob('C:/Users/user/Desktop/MS_DSBA/Electifs/AML/Demineur/Learning_dict_bandit/*')
        latest_file = max(list_of_files, key=os.path.getctime).replace("\\", "/")
        with open(latest_file) as f:
            self.learn_dict = json.load(f)

    def border(self, positions):
        """Return the position in the grid of a sub_state.
        """
        bord = [0, 0, 0, 0]
        for position in positions:
            if position[1] == 0:
                bord[1] = 1
            if position[1] == self.y_max:
                bord[0] = 1
            if position[0] == 0:
                bord[2] = 1
            if position[0] == self.x_max:
                bord[3] = 1
        return bord

    def sub_positions(self):
        """Split the grid into sub_grids which will correspond to sub_states.
        """
        for i in [4, 5]:
            for x in range(0, self.x_max - i + 2):
                for y in range(0, self.y_max - i + 2):
                    sub_square = []
                    for x2 in range(x, x + i):
                        for y2 in range(y, y + i):
                            sub_square.append((x2, y2))
                    self.sub_squares.append(sub_square)
        pass

    def sub_keys(self, square):
        bord = self.border(square)
        values = []
        for pos in square:
            values.append(self.dict_obs[pos])
        return [values, bord]

    def feed_dict(self, keys):
        for key in keys:
            switch = 0
            for value in key[0]:
                if value == 'X':
                    switch = 1
            if switch == 1:
                if key not in list(self.learn_dict.keys()):
                    self.learn_dict[repr(key)] = [[0] * len(key[0]), [0] * len(key[0])]
        pass

    def scoring(self, positions):
        scores = []
        for pos in positions:
            list_sub_state = []
            scores_pos = []
            for square in self.sub_squares:
                if pos in square:
                    list_sub_state.append([self.sub_keys(square), square.index(pos)])
            for i in list_sub_state:
                scores_pos.append(self.learn_dict[repr(i[0])][0][i[1]])
            scores.append(max(scores_pos))
        return scores

    def update_score(self, reward, key, position):
        switch = 0
        for value in key[0]:
            if value == 'X':
                switch = 1
        if switch == 1:
            last_avg_score = self.learn_dict[repr(key)][0][position]
            n_iter = self.learn_dict[repr(key)][1][position]
            self.learn_dict[repr(key)][0][position] = ((n_iter * last_avg_score) + reward) / (n_iter + 1)
            self.learn_dict[repr(key)][1][position] += 1
        pass

    def act(self, observation):
        """Acts given an observation of the environment.
        """
        self.turn += 1

        # At the beginning, the agent discover the grid and the values of x_max and y_max
        if self.game == 1 and self.turn == 1:
            for obs in observation:
                if obs[0][0] > self.x_max:
                    self.x_max = obs[0][0]
                if obs[0][1] > self.y_max:
                    self.y_max = obs[0][1]
            # Then he tag all the sub squares he will play with
            self.sub_positions()
            for dirpath, dirnames, files in os.walk('C:/Users/user/Desktop/MS_DSBA/Electifs/AML/Demineur/Learning_dict_bandit'):
                if files:
                    self.load_last_learn_dict()

        # From the last 100 games, the agent exploits
        if self.game == 901:
            self.epsilon = 0

        # Updating the dictionary of the observations + listing of the available positions to play with
        available_act = []
        for obs in observation:
            self.dict_obs[obs[0]] = obs[1]
            if obs[1] == 'X':
                available_act.append(obs[0])

        # Listing of the sub_states
        self.sub_state_keys = []
        for square in self.sub_squares:
            self.sub_state_keys.append(self.sub_keys(square))

        # Implementing new entries in the learn dictionary from the current sub_states
        self.feed_dict(self.sub_state_keys)

        # Epsilon greedy choice
        if np.random.random() < self.epsilon:
            ind = range(0, len(available_act))
            return available_act[np.random.choice(ind)]

        else:
            # Implementing the scores of each available position
            scores = self.scoring(available_act)
            best_score = max(scores)
            pos_best_score = []
            for i in range(0,len(scores)):
                if scores[i] == best_score:
                    pos_best_score.append(i)

            return available_act[np.random.choice(pos_best_score)]

    def reward(self, observation, action, reward):
        """Update the scores in the learning dictionary
        """
        for i in range(0, len(self.sub_squares)):
            if action in self.sub_squares[i]:
                square = self.sub_squares[i]
                self.update_score(reward, self.sub_state_keys[i], square.index(action))


class QAgent:
    def __init__(self,gridsize=(5,5)):
        """Init a new agent.
        """
        self.action_available=[]

        
        self.gridsize=gridsize
        self.numcases=self.gridsize[0]*self.gridsize[1]

        self.q = collections.defaultdict(lambda: np.zeros((self.numcases,), dtype=np.float32))
        self.gamma = 0.9
        self.t = 1
        self.eps=0.2
        
        self.game=1
        self.pending = None
        self.newgame = True


    def reset(self):
        """Reset the internal state of the agent, for a new run.

        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.

        You must **not** reset the learned parameters.
        """
        self.game+=1
        self.newgame = True

    def computeAvailableAction(self,observation):
        avail=[]
        for i,pos_scr in enumerate(observation):
            if pos_scr=="X":
                avail.append(i)
        return avail

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
        """
        s_list=[str(obs[1]) for obs in observation]
        s=" ".join(s_list)

        self.action_available=self.computeAvailableAction(s_list)

        if self.pending is not None:
            if self.newgame:
                s = "FINISH"
                self.newgame = False
            #else:
                #s = observation
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

        #s = observation
        # choose action
        #eps = 1.0/np.sqrt(self.t)s
        count_batch = self.game // 10000
        if (self.game - count_batch*10000)==9001:
            self.eps = 0
        if self.game % 10000 == 0:
            self.eps = 0.2

        if np.random.rand(0,1) > self.eps:
            #pos_index=np.random.randint(0,self.numcases)
            #pos = (pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            #return pos
            pos_index=random.choice(self.action_available)
            pos = (pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            return pos
        else:
            #pos_index= np.argmax(self.q[s])
            #pos=(pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            #return pos
            pos_index=None
            q_maxi=np.argsort(self.q[s])
            #print(q_maxi)
            #print(self.action_available)
            for i in range(len(q_maxi)):
                if q_maxi[len(q_maxi)-1-i] in self.action_available:
                    pos_index=q_maxi[len(q_maxi)-1-i]
                    break
            pos=(pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            return pos

   


    def reward(self, obs, act, reward):
        s_list=[str(o[1]) for o in obs]
        s=" ".join(s_list)
        self.pending = (s, act, reward)

####################################################CONVO NETWORK        





class QAgentConvoNetwork2:
    def __init__(self,gridsize=(5,5)):
        """Init a new agent.
        """
        self.action_available=[]

        self.numcases=gridsize[0]*gridsize[1]
        self.gridsize=gridsize

        self.MemorySize=1000 #Each time the container ExperienceReplay is full we fit the new data to our Conv neural net with SampleToFit observation
        self.SampleToFit=int(self.MemorySize*0.75) #70% of the data stored in ExperienceReplay is used for fitting the conv
        self.ExperienceReplay=[]
        self.batch_size=64
        self.epochs=10
        self.forgetRate=3.5/4 #the amount of infromation we choose to forget in the ExperienceReplay container

        self.gamma = 0.9
        self.t = 1
        
        self.q = collections.defaultdict(lambda: np.zeros((self.numcases,), dtype=np.float32))


        self.pending = None
        self.newgame = True
        self.game=1

        #self.model=None
        self.saveModel=False
        self.pursueT=True
        root_name='/Users/yaguethiam/Centrale_3A/AdvancedMachineLearning/FinalProject_Minesweeper/multipleAction/convo5x5_32_2_2_biss/'
        self.model_q_value_name_file=root_name+'convo_ql_5x5_convo_32_2_2_150000_q_value_records.json'
        self.model_convo_model_file_name=root_name+'convo_ql_5x5_convo_32_2_2_150000'
        
        if  self.pursueT:
            with open(self.model_q_value_name_file) as f:
                my_dict = json.load(f)
            for k in my_dict:
                self.q[k]=np.array(my_dict[k]).reshape(self.numcases,)
            self.model=loadModel(self.model_convo_model_file_name)
            
        else:
            self.model = Sequential()
            self.model.add(Conv2D(32, (2, 2), activation='relu',input_shape=(gridsize[0],gridsize[1],1)))
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
        if self.saveModel and self.game%25000==0:
            print("saving files")
            name=self.model_convo_model_file_name+"_"+str(self.game)
            saveModel(self.model,name,self.q)

        self.newgame = True
        self.game+=1

    def to_pixels(self,tt):
        image1=np.zeros(self.gridsize)
        
        for i in range(self.numcases):
            i_pos = (i // self.gridsize[1], i % self.gridsize[1])

            if tt[i]=="X":
                image1[i_pos[0],i_pos[1]]=255
            else:
                image1[i_pos[0],i_pos[1]]=float(tt[i])*10
        return image1/255.0

    def processTrainData(self,train):
        output=[]
        for tt in train:
            image1=self.to_pixels(tt[0])
            image2=self.to_pixels(tt[1])
            output.append(image2-image1)
        return np.array(output)

    def computeAvailableAction(self,observation):
        avail=[]
        for i,pos_scr in enumerate(observation):
            if pos_scr=="X":
                avail.append(i)
        return avail
            
    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
        """
        s_list=[str(obs[1]) for obs in observation]
        s=" ".join(s_list)

        self.action_available=self.computeAvailableAction(s_list)

        if self.t%self.MemorySize==0:
            #Train the model
            mybatch=random.sample(self.ExperienceReplay,self.SampleToFit)#[:self.SampleToFit]
            train_not_processed=[(myb[0],myb[1]) for myb in mybatch]
            train=self.processTrainData(train_not_processed)
            train=train.reshape(train.shape[0],self.gridsize[0],self.gridsize[1],1)
            targets=np.array([vec[2].reshape(self.gridsize[0]*self.gridsize[1],) for vec in mybatch])
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
            #s = observation
            self.t += 1
            (last_s, last_a, last_r) = self.pending
            pos_index = 0
            pos_index_temp = range(last_a[0]*self.gridsize[1], (last_a[0]+1)*self.gridsize[1])
            for i in pos_index_temp:
                if i%self.gridsize[1] == last_a[1]:
                    pos_index = i
                    break

            #print("----",self.q[last_s])
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
                self.q[last_s] = self.model.predict(to_add)[0]

            self.pending = None 

        #s = observations
        # choose action
        eps = 1.0/np.sqrt(self.t)
        if (np.random.rand(0,1) > eps and self.t<self.MemorySize):
            #pos_index=np.random.randint(0,self.numcases) 
            pos_index=random.choice(self.action_available)
            pos = (pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            return pos
        else:
            #pos_index= np.argmax(self.q[s])
            pos_index=None
            q_maxi=np.argsort(self.q[s])
            for i in range(len(q_maxi)):
                if q_maxi[len(q_maxi)-1-i] in self.action_available:
                    pos_index=q_maxi[len(q_maxi)-1-i]
                    break
           
            pos=(pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            return pos


    def reward(self, obs, act, reward):
        #self.pending = (obs, act, reward)
        s_list=[str(o[1]) for o in obs]
        s=" ".join(s_list)
        self.pending = (s, act, reward)


class QAgentConvoNetwork3:
    def __init__(self,gridsize=(5,5)):
        """Init a new agent.
        """
        self.action_available=[]

        self.numcases=gridsize[0]*gridsize[1]
        self.gridsize=gridsize

        self.MemorySize=10000 #Each time the container ExperienceReplay is full we fit the new data to our Conv neural net with SampleToFit observation
        self.SampleToFit=int(self.MemorySize*0.75) #70% of the data stored in ExperienceReplay is used for fitting the conv
        self.ExperienceReplay=[]
        self.batch_size=126
        self.epochs=15
        self.forgetRate=3.5/4 #the amount of infromation we choose to forget in the ExperienceReplay container

        self.gamma = 0.9
        self.t = 1
        self.eps = 0.2

        
        self.q = collections.defaultdict(lambda: np.zeros((self.numcases,), dtype=np.float32))


        self.pending = None
        self.newgame = True
        self.game=1

        #self.model=None
        self.saveModel=True #if sset to true, allow to save the model
        self.pursueT=False #run directly a loaded model when already exist
        root_name='/Users/yaguethiam/Centrale_3A/AdvancedMachineLearning/FinalProject_Minesweeper/multipleAction/'
        name_file='deepQ4x4'

        #self.model_q_value_name_file=root_name+'deepQ4x4'
        #self.model_convo_model_file_name=root_name+'convo5x5_8_2_2_2Layers_100000'

        self.model_convo_model_file_name=name_file

        if  self.pursueT:
            with open(self.model_q_value_name_file) as f:
                my_dict = json.load(f)
            for k in my_dict:
                self.q[k]=np.array(my_dict[k]).reshape(self.numcases,)
            self.model=loadModel(self.model_convo_model_file_name)
            
        else:
            self.model = Sequential()
            self.model.add(Conv2D(10, (2, 2), activation='relu',input_shape=(gridsize[0],gridsize[1],1)))
            self.model.add(Flatten())
            self.model.add(Dropout(0.3))            
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
        if self.saveModel and self.game%25000==0:
            print("saving files")
            name=self.model_convo_model_file_name+"_"+str(self.game)
            saveModel(self.model,name,self.q)

        self.newgame = True
        self.game+=1

    def to_pixels(self,tt):
        image1=np.zeros(self.gridsize)
        
        for i in range(self.numcases):
            i_pos = (i // self.gridsize[1], i % self.gridsize[1])

            if tt[i]=="X":
                image1[i_pos[0],i_pos[1]]=255
            else:
                image1[i_pos[0],i_pos[1]]=float(tt[i])*10
        return image1/255.0

    def processTrainData(self,train):
        output=[]
        for tt in train:
            image1=self.to_pixels(tt[0])
            image2=self.to_pixels(tt[1])
            output.append(image2-image1)
        return np.array(output)

    def computeAvailableAction(self,observation):
        avail=[]
        for i,pos_scr in enumerate(observation):
            if pos_scr=="X":
                avail.append(i)
        return avail
            
    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
        """
        s_list=[str(obs[1]) for obs in observation]
        s=" ".join(s_list)

        self.action_available=self.computeAvailableAction(s_list)

        if self.t%self.MemorySize==0:
            #Train the model
            mybatch=random.sample(self.ExperienceReplay,self.SampleToFit)#[:self.SampleToFit]
            train_not_processed=[(myb[0],myb[1]) for myb in mybatch]
            train=self.processTrainData(train_not_processed)
            train=train.reshape(train.shape[0],self.gridsize[0],self.gridsize[1],1)
            targets=np.array([vec[2].reshape(self.gridsize[0]*self.gridsize[1],) for vec in mybatch])
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
            #s = observation
            self.t += 1
            (last_s, last_a, last_r) = self.pending
            pos_index = 0
            pos_index_temp = range(last_a[0]*self.gridsize[1], (last_a[0]+1)*self.gridsize[1])
            for i in pos_index_temp:
                if i%self.gridsize[1] == last_a[1]:
                    pos_index = i
                    break

            #print("----",self.q[last_s])
            qsa = self.q[last_s][pos_index]
            target_qsa = last_r + self.gamma * self.q[s].max()
            target_qsa_vec=self.q[last_s]
            target_qsa_vec[pos_index]=target_qsa
            
            to_add=(last_s.split(' '),s.split(' '),target_qsa_vec)
            
            self.ExperienceReplay.append(to_add)
            to_add_not_processed=[(myb[0],myb[1]) for myb in [to_add]]
            to_add=self.processTrainData(to_add_not_processed)
            to_add=to_add.reshape(to_add.shape[0],self.gridsize[0],self.gridsize[1],1)

            #use the convo network to compute the update of the q-value
            #process to_add before prediction
            if self.t>self.MemorySize:
                
                self.q[last_s] = self.model.predict(to_add)[0]

            self.pending = None 

        #s = observations
        # choose action
        count_batch = self.game // 10000
        if (self.game - count_batch*10000)==9001:
            self.eps = 0
        if self.game % 10000 == 0:
            self.eps = 0.2

        if (np.random.rand(0,1) > self.eps or self.t<self.MemorySize):
            #pos_index=np.random.randint(0,self.numcases) 
            pos_index=random.choice(self.action_available)
            pos = (pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            return pos
        else:
            #pos_index= np.argmax(self.q[s])
            pos_index=None
            q_maxi=np.argsort(self.model.predict(to_add)[0])
            for i in range(len(q_maxi)):
                if q_maxi[len(q_maxi)-1-i] in self.action_available:
                    pos_index=q_maxi[len(q_maxi)-1-i]
                    break
           
            pos=(pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            return pos


    def reward(self, obs, act, reward):
        #self.pending = (obs, act, reward)
        s_list=[str(o[1]) for o in obs]
        s=" ".join(s_list)
        self.pending = (s, act, reward)


class QAgentConvoNetwork4:
    def __init__(self,gridsize=(5,5)):
        """Init a new agent.
        """
        self.action_available=[]

        self.numcases=gridsize[0]*gridsize[1]
        self.gridsize=gridsize

        self.MemorySize=10000 #Each time the container ExperienceReplay is full we fit the new data to our Conv neural net with SampleToFit observation
        self.SampleToFit=int(self.MemorySize*0.75) #70% of the data stored in ExperienceReplay is used for fitting the conv
        self.ExperienceReplay=[]
        self.batch_size=126
        self.epochs=15
        self.forgetRate=3.5/4 #the amount of infromation we choose to forget in the ExperienceReplay container

        self.gamma = 0.9
        self.t = 1
        self.eps = 0.2

        
        self.q = collections.defaultdict(lambda: np.zeros((self.numcases,), dtype=np.float32))


        self.pending = None
        self.newgame = True
        self.game=1

        #self.model=None
        self.saveModel=True
        self.pursueT=False
        root_name='/Users/yaguethiam/Centrale_3A/AdvancedMachineLearning/FinalProject_Minesweeper/multipleAction/'
        name_file='deepQ4x4'

       
        self.model_convo_model_file_name=name_file

        if  self.pursueT:
            with open(self.model_q_value_name_file) as f:
                my_dict = json.load(f)
            for k in my_dict:
                self.q[k]=np.array(my_dict[k]).reshape(self.numcases,)
            self.model=loadModel(self.model_convo_model_file_name)
            
        else:
            self.model = Sequential()
            self.model.add(Conv2D(10, (4, 4), activation='relu',input_shape=(gridsize[0],gridsize[1],1)))
            self.model.add(Flatten())
            self.model.add(Dropout(0.3))            
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
        if self.saveModel and self.game%25000==0:
            print("saving files")
            name=self.model_convo_model_file_name+"_"+str(self.game)
            saveModel(self.model,name,self.q)

        self.newgame = True
        self.game+=1

    def to_pixels(self,tt):
        image1=np.zeros(self.gridsize)
        
        for i in range(self.numcases):
            i_pos = (i // self.gridsize[1], i % self.gridsize[1])

            if tt[i]=="X":
                image1[i_pos[0],i_pos[1]]=255
            else:
                image1[i_pos[0],i_pos[1]]=float(tt[i])*10
        return image1/255.0

    def processTrainData(self,train):
        output=[]
        for tt in train:
            image1=self.to_pixels(tt[0])
            output.append(image1)
        return np.array(output)

    def computeAvailableAction(self,observation):
        avail=[]
        for i,pos_scr in enumerate(observation):
            if pos_scr=="X":
                avail.append(i)
        return avail
            
    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
        """
        s_list=[str(obs[1]) for obs in observation]
        s=" ".join(s_list)

        self.action_available=self.computeAvailableAction(s_list)

        if self.t%self.MemorySize==0:
            #Train the model
            mybatch=random.sample(self.ExperienceReplay,self.SampleToFit)#[:self.SampleToFit]
            train_not_processed=[(myb[0],myb[1]) for myb in mybatch]
            train=self.processTrainData(train_not_processed)
            train=train.reshape(train.shape[0],self.gridsize[0],self.gridsize[1],1)
            targets=np.array([vec[2].reshape(self.gridsize[0]*self.gridsize[1],) for vec in mybatch])
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
            #s = observation
            self.t += 1
            (last_s, last_a, last_r) = self.pending
            pos_index = 0
            pos_index_temp = range(last_a[0]*self.gridsize[1], (last_a[0]+1)*self.gridsize[1])
            for i in pos_index_temp:
                if i%self.gridsize[1] == last_a[1]:
                    pos_index = i
                    break

            #print("----",self.q[last_s])
            qsa = self.q[last_s][pos_index]
            target_qsa = last_r + self.gamma * self.q[s].max()
            target_qsa_vec=self.q[last_s]
            target_qsa_vec[pos_index]=target_qsa
            
            to_add=(last_s.split(' '),s.split(' '),target_qsa_vec)
            
            self.ExperienceReplay.append(to_add)
            to_add_not_processed=[(myb[0],myb[1]) for myb in [to_add]]
            to_add=self.processTrainData(to_add_not_processed)
            to_add=to_add.reshape(to_add.shape[0],self.gridsize[0],self.gridsize[1],1)

            #use the convo network to compute the update of the q-value
            #process to_add before prediction
            if self.t>self.MemorySize:
                
                self.q[last_s] = self.model.predict(to_add)[0]

            self.pending = None 

        #s = observations
        # choose action
        count_batch = self.game // 10000
        if (self.game - count_batch*10000)==9001:
            self.eps = 0
        if self.game % 10000 == 0:
            self.eps = 0.2

        if (np.random.rand(0,1) > self.eps or self.t<self.MemorySize):
            #pos_index=np.random.randint(0,self.numcases) 
            pos_index=random.choice(self.action_available)
            pos = (pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            return pos
        else:
            #pos_index= np.argmax(self.q[s])
            pos_index=None
            q_maxi=np.argsort(self.model.predict(to_add)[0])
            for i in range(len(q_maxi)):
                if q_maxi[len(q_maxi)-1-i] in self.action_available:
                    pos_index=q_maxi[len(q_maxi)-1-i]
                    break
           
            pos=(pos_index // self.gridsize[1], pos_index % self.gridsize[1])
            return pos


    def reward(self, obs, act, reward):
        #self.pending = (obs, act, reward)
        s_list=[str(o[1]) for o in obs]
        s=" ".join(s_list)
        self.pending = (s, act, reward)

Agent = QAgent
#Agent =RandomAgent
#Agent=QAgentConvoNetwork2 #take the difference of two images as input of the convolutional neural network
#Agent=QAgentConvoNetwork4 #take one image as input of the convolutional neural network
