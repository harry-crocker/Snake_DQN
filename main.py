from snake import Snake
from DQN import DeepQNetwork
import pygame
import random
import pickle
import time
'''
This is the main game loop to run, play and train snake snake
By default it will load DQN with saved weights and test

Keyboard inputs - Use 'Q' to quit running
                - Use 'T' to toggle training on and off
                - Use 'D' to toggle display on and off (faster training when off)
                - Use arrow keys as input if playing yourself

Arguments 

gamemode:       Integer
                0 = Human input, play the game using arrow keys
                1 = AI agent input

train:          Boolean
                True = DQN adds experiences to replay memory and trains
                False = Used to evaluate DQN performance without random movements

reset_weights:  Boolean
                True = reinitialise DQN and replay memory
                False = Load from saved file 'Saved_Data/Snake_Memory'

use_pygame:     Boolean
                True = Use Pygame (needed to see output and end game safely)
                False = Don't use Pygame (Used for training in google colab)

use_conv:       Boolean
                True = Use a Convolutional Neural Network for DQN (not yet functioning)
                False = Use regular MLP for DQN
'''

def main(gamemode=1, train=False, reset_weights=False, use_pygame=True, use_conv=False):
    start_time = time.time()
    # Initialise parameters
    run = True
    frame = 15 # Framerate

    instant_restart = True if gamemode==1 else False
    display = True

    # Window dimensions (Changes size of game board)
    win_width = 1000
    win_height = 600

    # Initialise Pygame
    if use_pygame:
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Snake")
        win = pygame.display.set_mode((win_width, win_height))
        clock = pygame.time.Clock()
        pygame.display.update()

    if use_conv:
        input_dims = 11
        num_actions = 4
    else:
        input_dims = 9
        num_actions = 3

    # Initialise loop objects
    snake = Snake(gamemode, win_width, win_height, use_conv)
    DQN = DeepQNetwork(input_dims, num_actions, reset_weights)
    action = 1
    mini_memory = []      
    scores = []   
    counter = 0     
    count_time = time.time()     
    percent_apple = 0
    percent_lose = 0   
    exploit = False
    explore = False

    print("Before loop --- %s seconds ---" % (time.time() - start_time))

    while run:
        # Outputs data while training 
        if counter % 50_000 == 5000 and train:
            rewards = [item[3] for item in DQN.memory]
            percent_apple = round(rewards.count(1) * 100/len(rewards), 1)
            percent_lose =  round(rewards.count(-1) * 100/len(rewards), 1)
            percent_closer =  round(rewards.count(0.05) * 100/len(rewards), 1)
            percent_further =  round(rewards.count(0) * 100/len(rewards), 1)
            print(f'Apples: {percent_apple}%, Loses: {percent_lose}%, Closer: {percent_closer}%, Further: {percent_further}%')
            print(f'Epsilon: {round(DQN.e, 3)}, Memory Length: {len(DQN.memory)}, Push Count: {DQN.push_count}')
            DQN.save_data()
            print('Time between prints: %s' % round(time.time() - count_time, 0))
            count_time = time.time()
        counter += 1

        if use_pygame:
            events = pygame.event.get()
            if display:
                clock.tick(frame)

        if snake.alive:
            # Get action depending on gamemode (player or DQN)
            if gamemode == 0:
                action = snake.player_input(events)
            elif gamemode == 1:
                action = DQN.get_action(snake.state, exploit, explore)

            snake.move(action)
            snake.update_board()
            if use_conv:
                snake.update_state_conv()
            else:
                snake.update_state()

            # Add a higer weight to experiences near death or reward (reducing sparse rewards)
            # Also maintain percentage for apples and deaths between thresholds to combat 'catastrophic forgetting'
            # Dont add to memory if died because stuck in cycle (move_counter > 1000)
            if train and snake.move_counter < 1000:
                experience = snake.get_experience()
                reward = experience[3]

                # if apple or death add events leading to this to replay memory
                if (reward == 1 and percent_apple < 8) or (reward == -1 and percent_lose < 8):
                    mini_memory.append(experience)
                    for exp in mini_memory[-8:]:
                        DQN.push_to_memory(exp)
                        DQN.train()
                    mini_memory = []
                # Also rarely add events randomly
                elif random.random() < 0.05 and min(percent_apple, percent_lose) > 5:
                    DQN.push_to_memory(experience)
                    DQN.train()
                else: 
                    if len(mini_memory) > 8:
                        mini_memory = []
                    mini_memory.append(experience)

        if display and use_pygame:
            snake.draw(win)
            pygame.display.update()

        # Restart game if snake dies (wait for R if player input)
        if not snake.alive:
            if gamemode == 0:   # Player 
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            # Re initiliase game
                            print(snake.score)
                            snake = Snake(gamemode, win_width, win_height, use_conv)
                            action = 1
            elif gamemode == 1: # DQN
                if (not train) or exploit:
                    scores.append(snake.score)
                    print(f'Score: {snake.score}    Average: {sum(scores)//len(scores)}')
                snake = Snake(gamemode, win_width, win_height, use_conv)
                action = 1
                if DQN.push_count > DQN.capacity:
                    if random.random() < 0.1:
                        exploit = True
                        explore = False
                    elif random.random() < 0.2:
                        exploit = False
                        explore = True
                    else:
                        exploit = False
                        explore = False

        if use_pygame:
            # Toggle between training and testing using T on keyboard
            # Toggle on and off display using D on keyboard (to improve speed)
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        train = False if train else True
                        print('Toggle Train')
                    if event.key == pygame.K_d:
                        display = False if display else True
                        print('Toggle Display')

            # End game using Q on keyboard or red cross
            for event in events: 
                if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                    if event.key == pygame.K_q or event.type == pygame.QUIT:
                        DQN.save_data()
                        path = 'Saved_Data'
                        filename = 'Scores'
                        object_to_save = scores
                        with open(f'{path}/{filename}', 'wb') as file:
                            pickle.dump(object_to_save, file)
                        run = False
                        quit()


if __name__ == "__main__":
    main(0, False, False, True, False)
