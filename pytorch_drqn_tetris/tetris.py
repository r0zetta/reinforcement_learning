import numpy as np
import random

class game:
    def __init__(self):
        self.state_width = 10
        self.state_height = 10
        self.replay = []
        self.reset()
        self.game_speed = 4
        self.min_speed = 2
        self.moves = ["left", "right", "down", "rotate_c", "rotate_a"]
        self.letters = ["l", "r", "d", "c", "a"]
        self.make_goal_state()
        self.num_actions = len(self.moves)

    def reset(self):
        self.state = np.zeros((self.state_width, self.state_height))
        self.board = np.zeros((self.state_width, self.state_height))
        self.get_new_piece()
        self.position = [random.randint(0,self.state_width-len(self.piece)), 0]
        self.timestep = 0
        self.set_state()

    def make_goal_state(self):
        self.goal_state = np.zeros((self.state_width, self.state_height))
        for x in range(self.state_width):
            for y in range(self.state_height-4, self.state_height):
                self.goal_state[x][y] = 1

    def print_state(self):
        return self.get_printable(self.state)

    def print_goal_state(self):
        return self.get_printable(self.goal_state)

    def get_printable(self, s):
        printable = ""
        for index, row in enumerate(s.T):
            printable += "    |" + "".join(["*" if x==1 else " " for x in row]) + "|   "
            printable += "".join([str(int(x)) for x in row]) + "\n"
        printable += "      " + "-"*(self.state_width-2) + "\n"
        return printable

    def set_state(self):
        self.state = np.copy(self.board)
        s_x = len(self.piece)
        s_y = len(self.piece[0])
        x = self.position[0]
        y = self.position[1]
        for i in range(s_x):
            for j in range(s_y):
                if self.piece[i][j] == 1:
                    self.state[x+i][y+j] = self.piece[i][j]

    def get_new_piece(self):
        pieces = [  [[1], [1], [1], [1]],
                    [[1, 0], [1, 0], [1, 1]],
                    [[0, 1], [0, 1], [1, 1]],
                    [[0, 1], [1, 1], [1, 0]],
                    [[1, 1], [1, 1]] ]
        self.piece = np.array(random.choice(pieces))

    def is_game_over(self):
        if not self.can_move_down() and self.position[1] == 0:
            return True
        return False

    def settle_piece(self):
        s_x = len(self.piece)
        s_y = len(self.piece[0])
        x = self.position[0]
        y = self.position[1]
        height_reached = y+s_y
        for i in range(s_x):
            for j in range(s_y):
                if self.piece[i][j] == 1:
                    self.board[x+i][y+j] = self.piece[i][j]
        return height_reached

    def remove_rows(self):
        rows_removed = 0
        temp = self.board.tolist()
        for index, row in enumerate(temp):
            if np.sum(row) == self.state_width:
                del temp[index]
                temp.insert(0, np.zeros((self.state_width)))
                rows_removed += 1
        if rows_removed > 0:
            self.board = np.array(temp)
        return rows_removed

    def overlap_check(self, pos, piece):
        x = pos[0]
        y = pos[1]
        s_x = len(piece)
        s_y = len(piece[0])
        clipping = self.board[x:x+s_x,y:y+s_y]
        if piece.shape != clipping.shape:
            return False
        for xp in range(s_x):
            for yp in range(s_y):
                if piece[xp][yp] == 1 and clipping[xp][yp] == 1:
                    return False
        return True

    def move_left(self):
        new_x = self.position[0]
        if new_x > 0:
            new_x -= 1
        return [new_x, self.position[1]]

    def move_right(self):
        new_x = self.position[0]
        if new_x <= self.state_width - len(self.piece[0]):
            new_x += 1
        return [new_x, self.position[1]]

    def move_down(self):
        new_y = self.position[1]
        if new_y <= self.state_height - len(self.piece[0]):
            new_y += 1
        return [self.position[0], new_y]

    def rotate(self, n):
        return np.rot90(self.piece, n)

    def rotate_c(self):
        return np.rot90(self.piece)

    def rotate_a(self):
        return np.rot90(self.piece, 3)

    def can_move_left(self):
        return self.overlap_check(self.move_left(), self.piece)

    def can_move_right(self):
        if self.position[0] < 1:
            return False
        return self.overlap_check(self.move_right(), self.piece)

    def can_move_down(self):
        height = len(self.piece[0])
        if self.position[1] + height >= self.state_height:
            return False
        return self.overlap_check(self.move_down(), self.piece)

    def can_rotate_a(self):
        new_shape = self.rotate_a()
        new_width = len(new_shape)
        if self.position[0] + new_width > self.state_width:
            return False
        new_height = len(new_shape[0])
        if self.position[1] + new_height > self.state_height:
            return False
        return self.overlap_check(self.position, self.rotate_a())

    def can_rotate_c(self):
        new_shape = self.rotate_c()
        new_width = len(new_shape)
        if self.position[0] + new_width > self.state_width:
            return False
        new_height = len(new_shape[0])
        if self.position[1] + new_height > self.state_height:
            return False
        return self.overlap_check(self.position, self.rotate_c())

    def sample_random_action(self):
        return random.randint(0, len(self.moves)-1)
    
    # This is a work in progress that I haven't had time to finish.
    # It does't do anything meaningful right now
    def move_ai(self):
        scan_x = 0
        scan_y = self.state_height
        found_fit = False
        desired_rotation = 0
        while found_fit == False:
            rotations = []
            upper_pos = []
            for n in range(0, 4):
                rot = self.rotate(n)
                y_pos = scan_y -len(rot)-1
                upper_pos.append(y_pos)
                rotations.append(rot)
            fit_rotations = []
            for index, r in enumerate(rotations):
                overlap = self.overlap_check([scan_x, upper_pos[index]], r)
                if overlap == True:
                    fit_rotations.append(index)
            if len(fit_rotations) > 0:
                sbcounts = []
                for f in fit_rotations:
                    bottom_row = np.copy(self.board[scan_y-1])
                    block = np.copy(rotations[f])
                    bottom_block = block[len(block)-1]
                    for p, b in enumerate(bottom_block):
                        if b == 1:
                            bottom_row[p] = 1
                    sbcounts.append(np.sum(bottom_row))
                if len(sbcounts) > 0:
                    desired_rotation = fit_rotations[np.argmax(sbcounts)]
                    found_fit = True
            if found_fit == False:
                scan_x += 1
                if scan_x + len(self.piece) > self.state_width:
                    scan_x = 0
                    scan_y -= 1
                if scan_y < 1:
                    return random.randint(0, len(self.moves)-1)
        if desired_rotation > 0:
            return 3
        elif self.position[0] < scan_x:
            return 1
        elif self.position[0] > scan_x:
            return 0
        elif self.position[0] == scan_x and self.position[0] < self.state_height-1:
            return 2

    def start_new_piece(self):
        best_height = self.settle_piece()
        rows_removed = self.remove_rows()
        self.get_new_piece()
        self.position = [random.randint(0,self.state_width-len(self.piece)), 0]
        return rows_removed, best_height

    def step(self, action):
        info = ""
        info += "\n    Action: " + self.moves[action]
        info += "\n    Piece height: " + str(len(self.piece[0]))
        info += "\n    Piece width: " + str(len(self.piece))
        if self.is_game_over():
            return self.state, -1, True, "Game over"
        self.timestep += 1
        invalid_action = False
        rows_removed = 0
        best_height = 0
        info += "\n    Old position: " + str(self.position)
        if action == self.moves.index("left"):
            if self.can_move_left():
                self.position = self.move_left()
            else:
                invalid_action = True
        elif action == self.moves.index("right"):
            if self.can_move_right():
                self.position = self.move_right()
            else:
                invalid_action = True
        elif action == self.moves.index("down"):
            if self.can_move_down():
                self.position = self.move_down()
            else:
                info += "\n    Settled piece"
                rows_removed, best_height = self.start_new_piece()
        elif action == self.moves.index("rotate_a"):
            if self.can_rotate_a():
                self.piece = self.rotate_a()
            else:
                invalid_action = True
        elif action == self.moves.index("rotate_c"):
            if self.can_rotate_c():
                self.piece = self.rotate_c()
            else:
                invalid_action = True
        info += "\n    New position: " + str(self.position)

        if self.timestep % self.game_speed == 0 and self.timestep > 0:
            if self.can_move_down():
                self.position = self.move_down()
            else:
                rows_removed, best_height = self.start_new_piece()
        reward = 0
        if invalid_action == True:
            info += "\n    Invalid action"
            #reward = -0.04
        #if best_height > 0:
            #reward = best_height/20
        if rows_removed > 0:
            reward = rows_removed * rows_removed
        info += "\n    Rows removed: " + str(rows_removed)
        info += "\n    Reward: " + str(reward)
        self.set_state()
        return self.state, reward, False, info

