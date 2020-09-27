import random
import copy
import numpy as np
import pygame
import math


class Snake:
    def __init__(self, gamemode, win_width, win_height, conv):
        self.win_width = win_width
        self.win_height = win_height
        self.block_size = 30
        self.border = 40
        self.game_width_blocks = (self.win_width - self.border*2)//self.block_size
        self.game_height_blocks = (self.win_height - self.border*2)//self.block_size
        self.game_width = self.game_width_blocks*self.block_size
        self.game_height = self.game_height_blocks*self.block_size
        self.game_x = (self.win_width - self.game_width)//2
        self.game_y = (self.win_height - self.game_height)//2 + self.border//2
        self.gamemode = gamemode
        self.init_fonts = False
        self.c1 = (73, 181, 49) # snake colours
        self.c2 = (53, 151, 49)
        self.pos = [random.randint(2, self.game_width_blocks-1), random.randint(0, self.game_height_blocks-1)]   # x, y
        self.direction = 1  # Global direction, 0 = north, 1 = east, 2 = south, 3 = west
        b = copy.deepcopy(self.pos)
        b1 = [b[0]-1, b[1]]
        b2 = [b[0]-2, b[1]]
        self.body = [b2, b1, b]    # Head is at the end of the list
        self.apple = [random.randint(0, self.game_width_blocks-1), random.randint(0, self.game_height_blocks-1)]
        self.score = 0
        self.alive = True
        self.flash = 0
        self.reward = 0
        self.board_border = 5 + 2   # CNN view radius plus 2
        self.board = -1*np.ones([self.game_width_blocks+self.board_border*2, self.game_height_blocks+self.board_border*2])
        self.update_board()
        self.apple_distance = self.get_distance_to_apple()
        self.move_counter = 0
        self.conv = conv
        if not conv:
            self.update_state()
        else:
            self.update_state_conv()
        self.previous_state = self.state


    def move(self, action):
        # Store action for experience replay
        self.action = action

        # The input to the conv is global image so the output should be too
        if self.conv:
            self.direction = action
        else:
            # Action will be left (0), continue (1) and right (2) from NN
            action -= 1
            # Changes action to: left (-1), right (1) or continue (0)
            # Need to change global direction of snake
            self.direction += action
            if self.direction == -1:
                self.direction = 3
            elif self.direction == 4:
                self.direction = 0

        # Update position
        if self.alive:
            if self.direction == 0:
                self.pos[1] -=1     # Move north is reducing y co-ord
            elif self.direction == 1:
                self.pos[0] +=1
            elif self.direction == 2:
                self.pos[1] +=1
            elif self.direction == 3:
                self.pos[0] -=1

            # Check for apple
            if self.pos == self.apple:
                keep_looping = True
                while keep_looping: # Wait for break
                    keep_looping = False
                    new_apple_pos = [random.randint(0, self.game_width_blocks-1), random.randint(0, self.game_height_blocks-1)]
                    if new_apple_pos in self.body:
                        keep_looping = True

                self.apple = new_apple_pos 
                self.reward = 1
                self.move_counter = 0
                self.score += 1
            else:
                self.body.pop(0)    # Remove zeroth index (the end of tail)
                # Calc reward (give small reward for moving closer to apple)
                if not self.conv:
                    self.prev_apple_distance = self.apple_distance
                    self.apple_distance = self.get_distance_to_apple()
                    if self.apple_distance < self.prev_apple_distance: # If closer
                        self.reward = 0.05
                    else:
                        self.reward = 0
                else:
                    self.update_board()
                    self.update_state_conv()
                    if 1 in self.state:
                        self.reward = 0.05                            
                    else:
                        self.reward = 0

            # Check for collision with body
            for block in self.body:
                if block == self.pos:
                    self.alive = False
                    self.reward = -1

            # Check for collision with wall
            if not 0<=self.pos[0]<self.game_width_blocks or not 0<=self.pos[1]<self.game_height_blocks:
                self.alive = False
                self.reward = -1

            self.body.append(copy.copy(self.pos))   # Add new position to body list
            self.move_counter += 1
            if self.move_counter > 1000:
                self.alive = False

    def update_board(self):
        # Reset board
        b = self.board_border
        self.board[b:-b,b:-b] = 0
        for block in self.body:
            x = block[0] + b
            y = block[1] + b
            self.board[x, y] = -1

        x = self.apple[0] + b
        y = self.apple[1] + b
        self.board[x, y] = 1

    def get_distance_to_apple(self):
        dx = abs(self.pos[0] - self.apple[0])
        dy = abs(self.pos[1] - self.apple[1])
        distance = dx + dy
        return distance


    def get_experience(self):
        experience = [self.previous_state, self.action, self.state, self.reward, self.alive]
        self.previous_state = self.state
        return experience

    def update_state_conv(self):
        # For use when using CNN 
        # Takes a picutre of the board radius # blocks around snakes head and uses that as state
        # 11x11 image
        x = self.pos[0] + self.board_border
        y = self.pos[1] + self.board_border
        radius = self.board_border - 2
        x_min = x - radius
        x_max = x + radius
        y_min = y - radius
        y_max = y + radius
        state = self.board[x_min:x_max+1, y_min:y_max+1]
        self.state = state

    def update_state(self):
        # The state is what the snake 'sees'
        # It see in 7 directions (no point seeing behind) plus angle to apple and snake length
        # The states (distance to objects) will be calculated globally then 
        # reordered to suit the local decision of the snake
        x = self.pos[0] + self.board_border
        y = self.pos[1] + self.board_border
        state = []
        inc_directions = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]]
        # Global: [N, NE, E, SE, S, SW, W, NW]
        for inc_i, inc_j in inc_directions:
            i = 0
            j = 0
            while True:     # Wait for break
                i += inc_i  # Value to increment i by to determine search direction
                j += inc_j
                block_value = self.board[x + i, y + j]
                if block_value != 0:
                    distance = max([abs(i), abs(j)])
                    if distance > 20: 
                        distance = 20
                    state_value = block_value*(1 - distance/20)
                    state.append(state_value)
                    break

        # Convert state list from global to local
        # Local: [F, FR, R, BR, B, BL, L, FL]
        # Shuffle list by append to end then delete 1st
        for _ in range(self.direction*2):
            state.append(state[0])
            state.pop(0)

        # In this order, B can easily be removed
        state.pop(4)

        # Calculate angle to apple
        # State is same as action directions
        # +ve clockwise from ahead
        dx = self.apple[0] - self.pos[0]
        dy = self.pos[1] - self.apple[1]
        glob_angle = 2*math.atan2(dx, dy)/math.pi    # where 90 deg = 1
        locl_angle = glob_angle - self.direction
        if locl_angle < -2:
            locl_angle += 4
        # apple_direc is continous between -1 and 1
        apple_direc = locl_angle/2
        state.append(apple_direc)

        # Append the normalised body length
        state.append(len(self.body)/100)
        self.state = state


    def player_input(self, events):
        # Only used if gamemode == 0
        direction = self.direction
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    direction = 0
                if event.key == pygame.K_RIGHT:
                    direction = 1
                if event.key == pygame.K_DOWN:
                    direction = 2
                if event.key == pygame.K_LEFT:
                    direction = 3

        previous_direction = self.direction
        if (previous_direction + direction) % 2 == 1:   # Sum is odd
            diff = previous_direction - direction
            if diff == 3 or diff == -1:
                action = 1 # right
            if diff == -3 or diff == 1:
                action = -1 # left
        else:
            action = 0  # Continue

        action += 1 # To match NN output
        return action
        

    def draw(self, win):
        # Draw backgrounds
        grey = (49, 61, 78)
        blue = (39, 50, 62)
        black = (70, 70, 70)
        white = (240, 240, 240)
        red = (255, 10, 10)
        block_size = self.block_size
        if not self.init_fonts:
            self.FONT = pygame.font.SysFont("comicsans", 40)
            self.FONT_BIG = pygame.font.SysFont("comicsans", 90)
            self.init_fonts = True
        # Background colour
        pygame.draw.rect(win, black, (0, 0, self.win_width, self.win_height))
        # Draw outline
        o_w = 4 # Outline width
        pygame.draw.rect(win, white, (self.game_x - o_w, self.game_y - o_w, self.game_width + 2*o_w, self.game_height + 2*o_w))
        # Chequer board
        i = 0
        for x in range(self.game_width_blocks):
            # Uncomment below to change chequer from lines
            # i += 1
            for y in range(self.game_height_blocks):
                colour = (grey if i % 2 ==0 else blue)
                pygame.draw.rect(win, colour, (self.game_x+x*block_size, self.game_y+y*block_size, block_size, block_size))
                i += 1

        # Draw snake
        for i, block in enumerate(self.body):
            x = block[0]
            y = block[1]

            if i == len(self.body) - 1:
                colour = self.c2
            else:
                colour = self.c1
            pygame.draw.rect(win, colour, (self.game_x+x*block_size, self.game_y+y*block_size, block_size, block_size))

        # Draw apple
        x = self.apple[0]
        y = self.apple[1]
        pygame.draw.rect(win, red, (self.game_x+x*block_size, self.game_y+y*block_size, block_size, block_size))
        # Score text
        text = self.FONT.render(f"Score: {self.score}", 1, white)
        win.blit(text, (45, 20))
        text = self.FONT.render(f"Press Q to quit", 1, white)
        win.blit(text, (self.win_width - 250, 20))

        self.flash += 1
        if self.alive == False and not self.flash % 6 == 0:
            font_size1 = self.FONT_BIG.size('GAME OVER')
            text = self.FONT_BIG.render('GAME OVER', 1, red)
            x = (self.win_width - font_size1[0])//2
            y = (self.win_height - font_size1[1])//2
            win.blit(text, (x, y))

            font_size2 = self.FONT.size('Press R to restart')
            text = self.FONT.render('Press R to restart', 1, red)
            x = (self.win_width - font_size2[0])//2
            y = (self.win_height - font_size2[1])//2 + font_size1[1]
            win.blit(text, (x, y))
