import time,os,sys, pygame
import numpy as np
import tensorflow as tf
from pygame.locals import *
from keras import Sequential, layers
from keras.optimizers import Adam
from keras.layers import Dense
from collections import deque
from keras.models import load_model
pygame.init()


class DQN:
    def __init__(self, size = 5000, inputnumbers = 4, outputnumbers = 4):
        self.inputnumbers = inputnumbers
        self.outputnumbers = outputnumbers
        self.learning_rate = 0.001
        self.momentum = 0.95
        self.eps_min = 0.1
        self.eps_max = 1.0
        self.eps_decay_steps = 2000000
        self.replay_memory_size = size
        self.replay_memory = deque([], maxlen=size)
        n_steps = 4000000  # total number of training steps
        self.training_start = 10000  # start training after 10,000 game iterations
        self.training_interval = 4  # run a training step every 4 game iterations
        self.save_steps = 1000  # save the model every 1,000 training steps
        self.copy_steps = 10000  # copy online DQN to target DQN every 10,000 training steps
        self.discount_rate = 0.99
        # Skip the start of every game (it's just waiting time).
        self.skip_start = 90
        self.batch_size = 100
        self.iteration = 0  # game iterations
        self.done = True  # env needs to be reset
        self.recordReward = 0

        self.model = self.DQNmodel()

        
        return


    
    def stats(self):
        memory = list(self.replay_memory)
        sizeOfMemory = len(memory)
        self.recordReward += memory[-1][2]

        # print('Total memory: ', str(sizeOfMemory), ' , total rewards: ', str(self.recordReward))
        return

    def DQNmodel(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(self.inputnumbers,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.outputnumbers, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def sample_memories(self, batch_size):
        indices = np.random.permutation(len(self.replay_memory))[:batch_size]
        # state, action, reward, next_state, continue
        cols = [[], [], [], [], []]
        for idx in indices:
            memory = self.replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3].reshape(-1,1), cols[4].reshape(-1, 1))

    def epsilon_greedy(self, q_values, step):
        self.epsilon = max(self.eps_min, self.eps_max -
                           (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
        if np.random.rand() < self.epsilon:
            return np.random.randint(10)  # random action
        else:
            return np.argmax(q_values)  # optimal action


    


class RoboObstacle:
    def __init__(self, fps=50, storageSize = 5000, possibilities = 4, windowsize = (300,300), xlimit = 10, ylimit = 10):
        # set up the window
        self.windowsize = windowsize
        self.DISPLAYSURF = pygame.display.set_mode(self.windowsize, 0, 32)
        pygame.display.set_caption('RoboObstacle')
        # set up the colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

        self.dict_shapes = {}
        self.shapes_state = []

        self.Agent = DQN(size = storageSize, inputnumbers = possibilities, outputnumbers = possibilities+1)
        pygame.init()
        self.FPS = fps
        self.fpsClock = pygame.time.Clock()

        self.possibilities = possibilities
        self.updateBinariesKey = {}
        self.movement = []
        self.dimension = self.windowsize[0]/xlimit

    def trainDQN(self):
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            self.Agent.sample_memories(self.Agent.batch_size))
        

        next_q_values = self.Agent.model.predict(X_state_val)
        max_next_q_values = np.max(
            next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * self.Agent.discount_rate * max_next_q_values

        # Train the online DQN
        # print('fitting: ', len(X_state_val), ' datas')
        self.Agent.model.fit(X_state_val, tf.keras.utils.to_categorical(
            X_next_state_val, num_classes=self.Agent.outputnumbers), verbose=0)


        return True

    def show_n_shapes(self, n, y_location, generate_shapes = True, xlimit = 10):
        if generate_shapes == True:
            self.dict_shapes = {}
            self.shapes_state = []
            # n varies from 0 to 7
            x_poses = [i for i in range(xlimit)]
            for m in range(n):
                x_position = np.random.choice(x_poses, replace = False)
                self.shapes_state.append(x_position)
                x_location = x_position * self.dimension
                color = tuple([np.random.choice([i for i in range(255)]) for i in range(3)])

                self.dict_shapes[str(m)] = (color, x_location)


                pygame.draw.rect(self.DISPLAYSURF, color, (x_location, y_location, self.dimension, self.dimension))

            
        else:
            for i, j in self.dict_shapes.items():
                pygame.draw.rect(self.DISPLAYSURF, j[0], (j[1], y_location, self.dimension, self.dimension))
                
        return

    def display(self):
        pygame.display.update()
        self.fpsClock.tick(self.FPS)

        for event in pygame.event.get():

            if event.type == QUIT:
                # self.AgentA.model.save('models/AgentA.h5')
                pygame.quit()
                sys.exit()
        return

    def displayObstacles(self, ylimit = 10, xlimit = 10):
        for i in range(ylimit):
            number_of_obstacles = np.random.choice([i for i in range(xlimit)])
            location = (i * self.windowsize[0])/10
            self.show_n_shapes(number_of_obstacles, y_location = location, generate_shapes = True, xlimit = xlimit)
        
        return True

    def evaluate(self, state):
        obs, reward, done, info = '', False, True, 'failure'
        if state in self.shapes_state:
            reward = False
        else:
            reward = True
        return obs, reward, done, info 

    def step(self, state):
        # state is the interpretation of the neural network next coordinate (x_y coordinate location)
        # Clear previous displays
        location_x, location_y = state.split('_')

        # decide state of robot
        # State varies from 0 to 9
        # display the robot
        pygame.draw.rect(self.DISPLAYSURF, self.BLACK, (int(location_x), int(location_y), 30, 30))
        pygame.draw.rect(self.DISPLAYSURF, self.BLUE, (int(location_x), int(location_y), 25,25))
        pygame.draw.rect(self.DISPLAYSURF, self.RED, (int(location_x), int(location_y), 20, 20))
        pygame.draw.rect(self.DISPLAYSURF, self.WHITE, (int(location_x), int(location_y), 15,15))
        pygame.draw.rect(self.DISPLAYSURF, self.BLUE, (int(location_x), int(location_y), 10, 10))
        pygame.draw.rect(self.DISPLAYSURF, self.GREEN, (int(location_x), int(location_y), 5, 5))
        pygame.draw.rect(self.DISPLAYSURF, self.WHITE, (int(location_x), int(location_y), 3, 3))
        self.display()
        # time.sleep()

        obs, reward, done, info = self.evaluate(state)
        return obs, reward, done, info


    def binarize(self, current_coordinate):
        # current_coordinate must follow this format, x_y. 
        # where x is the x coordinate
        # where y is the y coordinate
        # coordinate = str(x) + '_'+ str(y)

        # degree of movement (left, right, front, back, stay_in_position)

        # 10 locations, modify from index, 4 - 10
        # 0 - left
        # 1 - right
        # 2 - Front
        # 3 - Back

        x, y = tuple(current_coordinate.split('_'))
        x, y = int(x), int(y)

        if y < 0:
            y = 30

        if x < 0:
            x = 30
    

        self.updateBinariesKey[0] = str(x) + '_'+ str(y - 30)
        self.updateBinariesKey[1] = str(x) + '_'+ str(y + 30)
        self.updateBinariesKey[2] = str(x - 30) + '_'+ str(y)
        self.updateBinariesKey[3] = str(x + 30) + '_'+ str(y)
        self.updateBinariesKey[4] = str(x) + '_'+ str(y) # remain in position


        front_movement = (str(x) + '_'+ str(y - 30)) in self.dict_shapes.keys()
        back_movement = (str(x) + '_'+ str(y + 30)) in self.dict_shapes.keys()
        left_movement = (str(x - 30) + '_'+ str(y)) in self.dict_shapes.keys()
        right_movement = (str(x + 30) + '_'+ str(y)) in self.dict_shapes.keys()
        
        if (front_movement == False) and (back_movement == False) and (left_movement== False) and (right_movement == False):
            static = True
        else:
            static = False

        binary = np.array([front_movement, back_movement, left_movement, right_movement]).reshape(-1, 1)
        return binary

    
    def interpret(self, prediction):
        # interpret the result: x_y
        value = np.argmax(prediction)
        output = self.updateBinariesKey[value]
        return output


    def startingCoordinate(self):
        x, y = 0, 0
        m= int(self.windowsize[0]/self.dimension)
        choice = np.random.choice([i for i in range(m)])
        for i in range(choice, m):
            for j in range(int(10)):
                key = str(int(i*30)) + '_' + str(int(j*30))
                if key in self.dict_shapes.keys():
                    pass
                else:
                    x = int(i*30)
                    y = int(j*30)
                    break

            break
        return str(x) + '_' + str(y), (x, y) #key, coordinate


    def TrackNextMove(self, move):
        self.movement.append(move)
        output = False
        counts = np.unique(self.movement, return_counts = True)
        if counts[0][0] == self.updateBinariesKey[4] and counts[1][0] > 2000:
            output = True
        else:
            output = False
        return output




def TrainNetwork(iterations = 5000, model_name = 'RoboNET'):
    robo = RoboObstacle(storageSize = iterations)
    robo.DISPLAYSURF.fill(robo.WHITE)
    robo.displayObstacles()
    i = 0
    state = 0
    iteration = 0
    iterations = 2000000
    successes = 0
    key, coordinate = robo.startingCoordinate()
    while iteration < iterations:
        # Make move
        obs, reward, done, info = robo.step(key)

        # binarize Obstacles
        binaries = np.array(robo.binarize(key)).reshape(1, -1)
        predictions = robo.Agent.model.predict(binaries)
        print(binaries)
        action = np.argmax(predictions)
        key = robo.interpret(predictions)
        # Track next move, if next move is static for 2000 iterations break
        robo.TrackNextMove(key)
        # action = robo.Agent.epsilon_greedy(q_value, iteration)
        successes+= reward


        # Let's memorize what just happened
        baseline_value = robo.Agent.replay_memory_size
        # if successes > int(baseline_value/2) and iteration == len(list(robo.Agent.replay_memory)):
        if successes == int(baseline_value) and reward == True:
            robo.Agent.model.save('./models/{}.h5'.format(model_name))
            break
        elif successes < int(baseline_value * 0.8) and reward == True:
            # replace random false with new inputs
            robo.Agent.replay_memory.append((np.array(binaries[0]), action, reward, action, 1.0 - done))
        
        else:
            pass
        

        # robo.Agent.stats()

        # Train Network
        robo.trainDQN()
        i+= 1

        if i == 10:
            i = 0
        else:
            pass

        iteration += 1


        if 0xFF == ord('q'):
            break

           
        # display update
        robo.display()


    lengthOfQueue = len(list(robo.Agent.replay_memory))
    return successes, iteration, lengthOfQueue

if __name__ == "__main__":
    TrainNetwork()