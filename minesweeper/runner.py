"""
This is the machinnery that runs your agent in an environment.

This is not intented to be modified during the practical.
"""
import numpy as np
class Runner:
    def __init__(self, environment, agent, verbose=False):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose

        self.gameFinished=0
        self.gameLost=0
        self.maxIterReached=0

        self.averageTurnPerGame=[]
        self.averageCellDiscovered=[]

    def step(self):
        observation = self.environment.observe()
        action = self.agent.act(observation)
        (reward, stop) = self.environment.act(action)
        self.agent.reward(observation, action, reward)
        return (observation, action, reward, stop)

    def loop(self, games, max_iter):
        cumul_reward = 0.0

        for g in range(1, games+1):
            numStepBeforeBoom=0
            if (g - (g // 10000)*10000)== 9001:    
                cumul_reward = 0.0
                self.gameFinished=0
                self.gameLost=0
                self.maxIterReached=0
                self.averageTurnPerGame=[]

            self.agent.reset()
            self.environment.reset()
            for i in range(1, max_iter+1):
                numStepBeforeBoom+=1
                if self.verbose:
                    print("Simulation step {}:".format(i))
                    self.environment.display()
                (obs, act, rew, stop) = self.step()
                cumul_reward += rew
                if self.verbose:
                    print(" ->       observation: {}".format(obs))
                    print(" ->            action: {}".format(act))
                    print(" ->            reward: {}".format(rew))
                    print(" -> cumulative reward: {}".format(cumul_reward))
                    if stop is not None:
                        if stop=="End game":
                            print(" ->    You win a game!")
                            self.gameFinished+=1
                        elif stop=="Boom":
                            print(" ->    Boom!!!!!! and ... : {}".format(stop))
                            self.gameLost+=1
                if stop is not None:
                    if stop=="End game":
                        self.gameFinished+=1
                    elif stop=="Boom":
                        self.gameLost+=1
                if stop is not None:
                    break
                if i==max_iter:
                    self.maxIterReached+=1
            if self.verbose:
                print(" <=> Finished game number: {} <=>".format(g))
                print()
            self.averageTurnPerGame.append(numStepBeforeBoom)
            if g%10000==0:
                import csv
                report_fold_bandit = '/Users/yaguethiam/Centrale_3A/AdvancedMachineLearning/FinalProject_Minesweeper/multipleAction/saved_csv/'
                csvfile = str(report_fold_bandit + 'deePqlearning5x5_results.csv')
                output_values = [self.gameFinished, self.gameLost, np.mean(self.averageTurnPerGame),g,self.environment.gridsize]
                with open(csvfile, "a") as output:
                    writer = csv.writer(output, lineterminator='/')
                    for val in output_values:
                        writer.writerow([val])
                    writer.writerow('\n')
            
        #if self.verbose:
        print("Total number of game win: ", str(self.gameFinished))
        print("Total number of game lost: ", str(self.gameLost))
        print("Total number of game where max_iter reached : ", str(self.maxIterReached))
        print("Average turn before explosion : ", str(np.mean(self.averageTurnPerGame)))

            

        return cumul_reward

def iter_or_loopcall(o, count):
    if callable(o):
        return [ o() for _ in range(count) ]
    else:
        # must be iterable
        return list(iter(o))

class BatchRunner:
    """
    Runs several instances of the same RL problem in parallel
    and aggregates the results.
    """

    def __init__(self, env_maker, agent_maker, count, verbose=False):
        self.environments = iter_or_loopcall(env_maker, count)
        self.agents = iter_or_loopcall(agent_maker, count)
        assert(len(self.agents) == len(self.environments))
        self.verbose = verbose
        self.ended = [ False for _ in self.environments ]

        self.gameFinished=0
        self.gameLost=0

    def game(self, max_iter):
        rewards = []
        avsteps=[]
        for (agent, env) in zip(self.agents, self.environments):
            agent.reset()
            env.reset()
            game_reward = 0
            stepBD=0
            for i in range(1, max_iter+1):
                stepBD+=1
                observation = env.observe()
                action = agent.act(observation)
                (reward, stop) = env.act(action)
                agent.reward(observation, action, reward)
                game_reward += reward
                if stop is not None:
                    break
            rewards.append(game_reward)
            avsteps.append(stepBD)
        return sum(rewards)/len(rewards),sum(avsteps)/len(avsteps)

    def loop(self, games, max_iter):
        cum_avg_reward = 0.0
        averageStepsPerGameList=[]
        for g in range(1, games+1):
            if g == 99500 :
                cum_avg_reward = 0.0
                averageStepsPerGameList=[]
            avg_reward,averageStepsPerGame = self.game(max_iter)
            cum_avg_reward += avg_reward
            averageStepsPerGameList.append(averageStepsPerGame)
            if self.verbose:
                print("Simulation game {}:".format(g))
                print(" ->            average reward: {}".format(avg_reward))
                print(" -> cumulative average reward: {}".format(cum_avg_reward))
                print(" -> average step before explosion: {}".format(sum(averageStepsPerGameList)/len(averageStepsPerGameList)))
        return cum_avg_reward
