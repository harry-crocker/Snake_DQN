from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,Conv2D, MaxPooling2D, Flatten, Input
import math
import pickle
import random
import numpy as np
import time


class DeepQNetwork:
    def __init__(self, num_inputs, num_actions, reset_weights):
        self.num_inputs = num_inputs
        self.num_actions = num_actions    # number of possible actions
        if num_inputs < 10:
            self.model = self.create_model()
            self.target_model = self.create_model()
            self.conv = False
        else:
            self.model = self.create_model_conv()
            self.target_model = self.create_model_conv()
            self.conv = True

        print(self.model.summary())

        if reset_weights:
            self.memory = []
            self.push_count = 0
            self.train_loss = []
            self.q_preds = [[], [], [], []]
        else:
            weights, self.memory, self.push_count, self.train_loss, self.q_preds = self.load_data()
            self.model.set_weights(weights)
            
        print(f'Push Count:{self.push_count}')
        self.capacity = 300_000
        # e is epsilon, the probability that an action will be random
        self.e_start = 1    
        self.e_end = 0.1
        self.e_decay = 0.00001
        self.batch_size = 512
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0
        self.future_discount = 0.99
        self.update_target_every = 300
        self.test_states = []
       

    def create_model(self):
        model = Sequential()
        model.add(Input(shape=(self.num_inputs,)))
        model.add(Dense(64, 'relu'))
        model.add(Dense(128, 'relu'))
        model.add(Dense(256, 'relu'))
        model.add(Dense(256, 'relu'))
        model.add(Dense(128, 'relu'))
        model.add(Dense(64, 'relu'))
        model.add(Dense(self.num_actions))
        model.compile(loss="mse", optimizer='Adam', metrics=['accuracy'])
        return model

    def create_model_conv(self):
        model = Sequential()
        shape = (self.num_inputs, self.num_inputs, 1)
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=shape, padding='same'))
        # model.add(MaxPooling2D((2, 2), strides = 1))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # model.add(MaxPooling2D((2, 2)))
        # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_actions))
        model.compile(loss="mse", optimizer='Adam', metrics=['accuracy'])
        return model

    def push_to_memory(self, experience):
        # Saves an experience to memory
        self.push_count += 1
        if len(self.memory) < self.capacity:
            # Append directly to memory
            self.memory.append(experience)
        else: 
            # Add new experience to memory, overwriting from start (1002 % 1000 = 2)
            i = self.push_count % self.capacity
            self.memory[i] = experience


    def get_action(self, state, exploit, explore):
        # Decay function starting at e_start and reaching asymptote at e_end
        self.e = self.e_end + (self.e_start - self.e_end) * \
                    math.exp(-1. * self.push_count * self.e_decay) 

        if (self.e > random.random() or explore) and not exploit:
            # Execute a random action to explore the environment
            action = random.randrange(self.num_actions)
            return action
        else: 
            # Use NN to predict action to exploit environment
            inputs = state
            if self.conv:
                inputs = np.reshape(state, (1, self.num_inputs, self.num_inputs, 1))
            else:
                inputs = np.array([state])
            # Get the maximum index of the output as the action
            outputs = self.model(inputs, training=False)[0]
            action = np.argmax(outputs)
            return action

    def train(self):
        # check if memory is large enough to take a batch
        if len(self.memory) < self.batch_size*2:
            return

        # Only train 20 times for every batch size added to replay memory
        if not self.push_count % (self.batch_size//20) == 1:
            return

        batch = random.sample(self.memory, self.batch_size)

        # Extract features from memory
        current_states =    np.array([item[0] for item in batch])
        actions =           np.array([item[1] for item in batch])
        new_states =        np.array([item[2] for item in batch])
        rewards =           np.array([item[3] for item in batch])
        alives =             np.array([item[4] for item in batch])

        if self.conv:
            current_states = np.expand_dims(current_states, axis=3)
            new_states = np.expand_dims(new_states, axis=3)

        # Get Q values (note different models)
        # Returns a list of lists with shape = (batch_size, num_actions)
        current_qs_list = self.model.predict(current_states)
        future_qs_list = self.target_model.predict(new_states)



        X = current_states
        Y = []

        for i in range(self.batch_size):
            if alives[i]:
                # Get the max q value from prediction 
                # This is the total future reward if the AI follows the policy until game ends
                max_future_q = np.max(future_qs_list[i])
                new_q = rewards[i] + self.future_discount*max_future_q
            else:
                # If the snake is dead then the max_future_q must be zero since no further 
                # rewards can be obtained
                new_q = rewards[i]

            # insert the new q value into training data
            # Eg current_qs = [1, 2, 3], new_q = 10, action = 0
            # insert new q at the index of the action
            # becomes >> [10, 2, 3] 
            current_qs = current_qs_list[i]
            current_qs[actions[i]] = new_q

            # Append to training data
            Y.append(current_qs)

        history = self.model.fit(X, np.array(Y), batch_size=self.batch_size, verbose=0, shuffle=False)
        self.train_loss.append(history.history['loss'])
        # Update target model every 
        self.target_update_counter += 1
        if self.target_update_counter > self.update_target_every:
            print('Update model')
            self.target_update_counter = 0
            self.target_model.set_weights(self.model.get_weights())


    def save_data(self):
        filename = 'Snake_Memory_Conv' if self.conv else 'Snake_Memory'
        object_to_save = [self.model.get_weights(), self.memory, self.push_count, self.train_loss, self.q_preds]
        with open(f'{filename}.pkl', 'wb') as file:
            pickle.dump(object_to_save, file)



    def load_data(self):
        filename = 'Snake_Memory.pkl'
        with open(filename, 'rb') as file:
            saved = pickle.load(file)
        return saved
        