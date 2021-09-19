import time,os,sys, pygame, cv2
import numpy as np
import tensorflow as tf
from pygame.locals import *
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from collections import deque
from tensorflow.keras.models import load_model
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
        self.replay_memory_size = size #number of experiences stored in replay memory
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
        '''This method keep statistics on total rewards obtained during an episode'''
        memory = list(self.replay_memory)
        sizeOfMemory = len(memory)
        self.recordReward += memory[-1][2]

        # print('Total memory: ', str(sizeOfMemory), ' , total rewards: ', str(self.recordReward))
        return 'Total memory: ', str(sizeOfMemory), ' , total rewards: ', str(self.recordReward)

    def DQNmodel(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(self.inputnumbers,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.outputnumbers, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def sample_memories(self, batch_size):
        '''This method defines how memories/experiences are sampled from the replay memory 
        before being passed as parameters to the DQN'''
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
        '''This method defines the exploration policy for the agent based on the epsilon
        greedy algorithm'''
        self.epsilon = max(self.eps_min, self.eps_max -
                           (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)  # random action
        else:
            return np.argmax(q_values)  # optimal action


    


class RoboObstacle:
    def __init__(self, fps=50, storageSize = 5000, possibilities = 4, windowsize = (300,300), xlimit = 10, ylimit = 10, env_learning_rate=0.3):
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
        #set number of possible grids on x and y axis
        self.xlimit = xlimit
        self.ylimit = ylimit

        self.dict_shapes = {} #dictionary holding position of obstacles per row in the window or environment
        self.shapes_state = []

        self.Agent = DQN(size = storageSize, inputnumbers = possibilities, outputnumbers = possibilities+1)
        pygame.init()
        self.FPS = fps
        self.fpsClock = pygame.time.Clock()

        self.possibilities = possibilities
        self.updateBinariesKey = {}
        self.movement = []
        self.dimension_x = int(self.windowsize[0]/xlimit) #width of each grid
        self.dimension_y = int(self.windowsize[1]/ylimit) #height of each grid
        self.startLocation = (0, 0)
        self.endLocation = (0, 0)
        self.Endkey = ''
        self.DecisionTracker = {}
        self.currentDecision = []

        #set factor for number of obstacles
        self.env_learning_rate = env_learning_rate

        #directory to image of destination robot
        self.RoboDir = './Images/dest.jpg'
        #set parameter for static robot
        self.static = False
        #end program
        self.end = False

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

    def show_n_shapes(self , generate_shapes = True):
        '''This method defines the part of the environment that has obstacles. It also enables retaining the obstacle
        positions within the environment when generate_shapes is set to False'''
        # display Robot
        # RobotImage = pygame.image.load(self.RoboDir)
        # RobotImage = pygame.transform.scale(RobotImage, (30, 30))
        # self.DISPLAYSURF.blit(RobotImage, self.endLocation)
        

        if generate_shapes == True:
            for i in range(self.ylimit):
                x_poses = [f for f in range(self.xlimit)] #list of all possible obstacle positions on the x-axis
                # number_of_obstacles = np.random.choice(x_poses)
                number_of_obstacles = int(self.xlimit * self.env_learning_rate) #number of obstacles per row
                y_location = (i * self.windowsize[1])/self.ylimit
                #location of obstacles
                self.shapes_state = []
                #iterate over each row and randomly place (and colour) m number of obstacles and store all selected grid positions in self.dict_shapes 
                for m in range(number_of_obstacles):
                    x_position = np.random.choice(x_poses, replace = False)
                    self.shapes_state.append(x_position)
                    x_location = x_position * self.dimension_x
                    color = tuple([np.random.choice([rgb for rgb in range(255)]) for i in range(3)])

                    self.dict_shapes[str(int(x_location))+'_'+str(int(y_location))] = (color, x_location, y_location)


                    pygame.draw.rect(self.DISPLAYSURF, color, (x_location, y_location, self.dimension_x, self.dimension_y))

        #retain the environment structure when generate_shapes is set to False    
        else:
            for i, j in self.dict_shapes.items():
                pygame.draw.rect(self.DISPLAYSURF, j[0], (j[1], j[2], self.dimension_x, self.dimension_y))
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

    def displayObstacles(self, generate_shapes = True):
        if generate_shapes == True:
            # ylimit = 10,  xlimit = 10, generate_shapes = True
            self.show_n_shapes(generate_shapes = generate_shapes)
        else:
            self.show_n_shapes(generate_shapes = generate_shapes)
       
        return True

    def evaluate(self, state):
        obs, reward, done, info = '', False, True, 'failure'
        if state in self.shapes_state:
            reward = False
        else:
            reward = True
        return obs, reward, done, info 

    def DrawRobot(self, location_x, location_y):
        box_dimension_x, box_dimension_y = int(self.windowsize[0]/self.xlimit), int(self.windowsize[1]/self.ylimit)

        self.robot = pygame.image.load('Images/robot.png')
        self.robot = pygame.transform.scale(self.robot, (box_dimension_x, box_dimension_y))
        
        self.DISPLAYSURF.blit(self.robot, (int(location_x), int(location_y)))
                
        return
        

    def step(self, state):
        # state is the interpretation of the neural network next coordinate (x_y coordinate location)
        # Clear previous displays
        location_x, location_y = state.split('_')

        # decide state of robot
        # State varies from 0 to 9
        # display the source robot
        self.DrawRobot(location_x, location_y)

        # display the destination robot
        xdest, ydest = self.endLocation
        
        self.DrawRobot(xdest, ydest)
        self.display()
        # time.sleep()

        obs, reward, done, info = self.evaluate(state)
        return obs, reward, done, info


    def binarize(self, current_coordinate):
        '''This method converts movements to coordinates and updates the values of self.updateBinariesKey keys.
        It also makes sure the agent never goes out of the window frame. It checks if the chosen movement is an obstacle 
        by comparing with self.dict_shapes, a dictionary that stores obstacle positions'''
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
        # print(x, y)

        if y < 0:
            y = self.dimension_y

        if y > self.windowsize[1]:
            y =self.windowsize[1] - self.dimension_y

        if x < 0:
            x = self.dimension_x

        if x > self.windowsize[0]:
            x =self.windowsize[0] - self.dimension_x
    

        self.updateBinariesKey[0] = str(x) + '_'+ str(y - self.dimension_y)
        self.updateBinariesKey[1] = str(x) + '_'+ str(y + self.dimension_y)
        self.updateBinariesKey[2] = str(x - self.dimension_x) + '_'+ str(y)
        self.updateBinariesKey[3] = str(x + self.dimension_x) + '_'+ str(y)
        self.updateBinariesKey[4] = str(x) + '_'+ str(y) # remain in position

        front_movement = (str(x) + '_'+ str(y - self.dimension_y)) in self.dict_shapes.keys()
        back_movement = (str(x) + '_'+ str(y + self.dimension_y)) in self.dict_shapes.keys()
        left_movement = (str(x - self.dimension_x) + '_'+ str(y)) in self.dict_shapes.keys()
        right_movement = (str(x + self.dimension_x) + '_'+ str(y)) in self.dict_shapes.keys()
        
        if self.Euclidean_distance(current_coordinate, self.Endkey) == 0:
            print('Arrived destination')
            time.sleep(5)
            self.end = True
            
        if (front_movement == False) and (back_movement == False) and (left_movement== False) and (right_movement == False):
            self.static = True
            # self.end = True

        else:
            self.static = False

        binary = np.array([front_movement, back_movement, left_movement, right_movement]).reshape(-1, 1)
        return binary

    
    def interpret(self, prediction):
        # interpret the result: x_y
        value = np.argmax(prediction)
        output = self.updateBinariesKey[value]
        return output


    def DecideCoordinate(self, excludingPoint = None):
        '''This method randomly selects the start and end coordinates of the agent
        '''
        x, y = 0, 0
        m= int(self.windowsize[0]/self.dimension_x)
        choice = np.random.choice([i for i in range(m)])
        for i in range(choice, m):
            for j in range(int(self.ylimit)):
                key = str(int(i*self.dimension_x)) + '_' + str(int(j*self.dimension_y))
                if key in self.dict_shapes.keys() or key == excludingPoint:
                    pass
                else:
                    x = int(i*self.dimension_x)
                    y = int(j*self.dimension_y)
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
    
    def Euclidean_distance(self, source, destination):
        x1, y1 = tuple(source.split('_'))
        x1, y1 = int(x1), int(y1)
        x2, y2 = tuple(destination.split('_'))
        x2, y2 = int(x2), int(y2)
        distance = np.sqrt(pow(y2-y1, 2)+ pow(x2 - x1, 2))
        return distance

    def sortMovement(self, dictData):
        vals = list(dictData.values())
        keys = list(dictData.keys())
        vals.sort()
        sortedKeys = []
        for i in vals:
            for j in keys:
                if dictData[j] == i:
                    sortedKeys.append(j)

        return sortedKeys

        

    def DefineAction(self, key):
        x, y = tuple(key.split('_'))
        x, y = int(x), int(y)

        front_movement = (str(x) + '_'+ str(y - self.dimension_y)) in self.dict_shapes.keys()
        back_movement = (str(x) + '_'+ str(y + self.dimension_y)) in self.dict_shapes.keys()
        left_movement = (str(x - self.dimension_x) + '_'+ str(y)) in self.dict_shapes.keys()
        right_movement = (str(x + self.dimension_x) + '_'+ str(y)) in self.dict_shapes.keys()

        OutputDict = {}
        for i, BooleanStatus in enumerate([front_movement, back_movement, left_movement, right_movement]):
            if BooleanStatus == False:
                # calcuate Euclidean Distance
                 distance = self.Euclidean_distance(self.updateBinariesKey[i], self.Endkey)
                 OutputDict[i] = distance
            else:
                pass
        
        interpretPosition = {0: 'front', 1: 'back', 2: 'left', 3: 'right', 4: 'stay'}
        
        # Decide to go through the path with the shortest distance
        # print(OutputDict)
        decision = self.sortMovement(OutputDict)
        self.currentDecision = decision
        # print(decision[0], interpretPosition[decision[0]])
        return decision[0], interpretPosition[decision[0]]




def TrainNetwork(iterations = 5000, model_name = 'RoboNET'):
    robo = RoboObstacle(storageSize = iterations)
    robo.DISPLAYSURF.fill(robo.WHITE)
    
    
    #show robot on image
    # frame = cv2.imread('./Images/dest.jpg')
    # cv2.imshow('image', frame)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    robo.displayObstacles(generate_shapes = True)
    Endkey, EndCoordinate = robo.DecideCoordinate() # Ending Coordinate
    robo.endLocation = EndCoordinate
    robo.Endkey = Endkey
    
    i = 0
    state = 0
    iteration = 0
    iterations = 2000000
    successes = 0
    
    startKey, startCoordinate = robo.DecideCoordinate(excludingPoint = Endkey) # starting Coordinate
    robo.startLocation = startCoordinate
    key = startKey
    print('=============================================================================')
    print('Start Coordinate: ', startCoordinate, ' End Coordinate: ', EndCoordinate)
    print('=============================================================================')
    print('boundaries coordinates')
    print('=============================================================================')
    print(robo.dict_shapes)
    print('=============================================================================')
    while (not robo.end) and (iteration < iterations):
        # Fill window with white grids
        robo.DISPLAYSURF.fill(robo.WHITE)
        # Display Generated Obstacles
        robo.displayObstacles(generate_shapes = False)
        # Make move
        obs, reward, done, info = robo.step(key)

        # binarize Obstacles
        binaries = np.array(robo.binarize(key)).reshape(1, -1)
        predictions = robo.Agent.model.predict(binaries)
        
        # action = np.random.choice([i for i in range(5)])
        action, interpretAction = robo.DefineAction(key)
        # key = robo.interpret(predictions)
        key = robo.updateBinariesKey[action]
        
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
        
        # print('binaries: ', binaries, ' predictions: ', predictions, ' action: ', action, ' Key: ', key, ' Data Size: ', robo.Agent.stats())
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