#!/usr/bin/env python3

import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import PIL
import collections
import argparse


DEATH_REWARD = -1
FOOD_REWARD = 1
NOTHING_REWARD = 0

UP, LEFT, DOWN, RIGHT = range(4)
NOTHING, HEAD, SNAKE, FOOD, WALL = range(5)
_STEP_X = [0, -1, 0, 1]
_STEP_Y = [-1, 0, 1, 0]
COLORS = np.array([[200, 200, 200], [0, 200, 0], [0, 100, 0],
                   [250, 50, 50], [100, 50, 0]], dtype=np.uint8)


class Snake:
    def __init__(self, sz):
        self.world = np.empty((sz, sz), dtype=np.uint8)
        self.reset()

    def reset(self):
        sz = self.world.shape[0]
        self.world.fill(NOTHING)
        self.growth_timer = 4
        # Borders
        self.world[:, 0::sz-1] = WALL
        self.world[0::sz-1, :] = WALL
        # Snake
        hx = sz // 2
        hy = sz // 2
        self.snake = collections.deque([(hx, hy)])
        self.world[hy, hx] = HEAD
        # Apple
        self._place_apple()

    def _place_apple(self):
        sz = self.world.shape[0]
        while True:
            x = random.randrange(sz)
            y = random.randrange(sz)
            if self.world[y, x] == NOTHING:
                break
        self.world[y, x] = FOOD
        self.food = (x, y)

    def random_move(self):
        for _ in range(40):
            a = random.randrange(4)
            hx, hy = self.snake[0]
            hx += _STEP_X[a]
            hy += _STEP_Y[a]
            if self.world[hy, hx] in (NOTHING, FOOD):
                break
        return a

    def make_action(self, a):
        eaten = self._move_head(a)
        if eaten == FOOD:
            self._place_apple()
            self.growth_timer += 2
            return FOOD_REWARD
        elif eaten != NOTHING:
            self.reset()
            return DEATH_REWARD
        if self.growth_timer > 0:
            self.growth_timer -= 1
        else:
            self._move_tail()
        return NOTHING_REWARD

    def _move_tail(self):
        tx, ty = self.snake.pop()
        self.world[ty, tx] = NOTHING

    def _move_head(self, direction):
        hx, hy = self.snake[0]
        self.world[hy, hx] = SNAKE
        hx += _STEP_X[direction]
        hy += _STEP_Y[direction]
        ret = self.world[hy, hx]
        if ret == SNAKE:
            hx -= 2 * _STEP_X[direction]
            hy -= 2 * _STEP_Y[direction]
            ret = self.world[hy, hx]
        self.world[hy, hx] = HEAD
        self.snake.appendleft((hx, hy))
        return ret

    def image(self):
        return COLORS[self.world]

    def features(self):
        # 3x3 pixels around the head + direction for food
        hx, hy = self.snake[0]
        win = self.world[hy-1:hy+2, hx-1:hx+2]
        f = np.zeros((5, 9))
        f[win.ravel(), np.arange(9)] += 1
        foodx = self.food[0] - hx
        foody = self.food[1] - hy
        food = np.array([foody, foodx])
        dist = np.maximum(1, np.sqrt(foodx ** 2 + foody ** 2))
        return np.concatenate((f.ravel(), food / dist))


def q_learning(environment, actions, lr, steps, gamma, epsilon):
    """Train a linear agent using Q-learning."""    
    x = environment.features()
    W = np.zeros((actions, x.size))
    b = np.zeros(actions)
    q = W @ x + b
    for step in range(steps):
        if random.random() < epsilon:
            a = random.randrange(actions)
        else:
            a = q.argmax()
        r = environment.make_action(a)
        x_next = environment.features()
        q_next = W @ x_next + b
        update = -(r + gamma * q_next.max() - q[a])
        W[a, :] -= lr * update * x
        b[a] -= lr * update
        x = x_next
        q = q_next
    return W, b


def make_image(im):
    im2 = PIL.Image.fromarray(im)
    h = im.shape[0] * 16
    w = im.shape[1] * 16
    im2 = im2.convert("P", dither=PIL.Image.NONE)
    return im2.resize((w, h), PIL.Image.NEAREST)
    

def run_agent(environment, w, b, output, steps):
    total_reward = 0
    frames = []
    for step in range(1, steps + 1):
        scores = w @ environment.features() + b
        a = np.argmax(scores)
        r = environment.make_action(a)
        total_reward += r
        frames.append(make_image(environment.image()))
    frames[0].save(output, save_all=True, append_images=frames[1:])
    print("{} {:.3f}".format(step, total_reward / step))


def play_game():
    """Let the user play the game with the keyboard."""
    acts = {"j": LEFT, "l": RIGHT, "i": UP, "k": DOWN}
    score = 0

    def press(event):
        nonlocal score
        key = event.key.lower()
        if key == 'q':
            sys.exit()
        try:
            a = acts[key]
        except KeyError:
            return
        reward = snake.make_action(a)
        score += reward
        plt.title(str(score))
        plt.imshow(snake.image())
        plt.gcf().canvas.draw()
    snake = Snake(20)
    plt.gcf().canvas.mpl_connect('key_press_event', press)
    plt.xlabel("Use IJKL to play the game, Q to quit")
    plt.title(str(score))
    plt.imshow(snake.image())
    plt.show()


def _main():
    desc = "The game of Snake with Q-learning"
    parser = argparse.ArgumentParser(description=desc)
    a = parser.add_argument
    a("-a", "--agent", default="snake.npz", help="file with the parameters of the agent")
    commands = ["play", "train", "run"]
    a("command", default="play", choices=commands, help="program command")
    a("-s", "--steps", type=int, default=1000, help="training steps")
    a("-g", "--gamma", type=float, default=0.9, help="discount factor")
    a("-e", "--epsilon", type=float, default=0.01, help="randomness")
    a("-l", "--lr", type=float, default=0.01, help="learning rate")
    a("--save_every", type=int, default=100000, help="frequency of saves")
    a("-o", "--output", default="snake.gif", help="output animated image")
    args = parser.parse_args()

    if args.command == "play":
        play_game()
    elif args.command == "train":
        snake = Snake(20)
        w, b = q_learning(snake, 4, args.lr, args.steps, args.gamma, args.epsilon)
        if args.agent:
            np.savez(args.agent, w=w, b=b)
    else:
        snake = Snake(20)
        data = np.load(args.agent)
        run_agent(snake, data["w"], data["b"], args.output, args.steps)


if __name__ == "__main__":
    _main()
