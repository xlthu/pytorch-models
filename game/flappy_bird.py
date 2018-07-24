import random
import os.path
import sys
from itertools import cycle

import numpy as np
from .rect import Rect

import PIL.Image
import PIL.ImageDraw

__all__ = ["init", "SCREENWIDTH", "SCREENHEIGHT", "SCREEN", "Flappy"]

_ROOT = os.path.dirname(os.path.abspath(__file__))

def _r(path):
    return os.path.join(_ROOT, path)

# path of player with different states
PLAYER_PATH = (
    _r('assets/sprites/redbird-upflap.png'),
    _r('assets/sprites/redbird-midflap.png'),
    _r('assets/sprites/redbird-downflap.png')
)

# path of background
BASE_PATH = _r('assets/sprites/base.png')
BACKGROUND_PATH = _r('assets/sprites/background-black.png')

# path of pipe
PIPE_PATH = _r('assets/sprites/pipe-green.png')

def init(display=True):
    global FPS, SCREENWIDTH, SCREENHEIGHT, FPSCLOCK, SCREEN, IMAGES, \
            HITMASKS, PIPEGAPSIZE, BASEY, PLAYER_WIDTH, PLAYER_HEIGHT, \
            PIPE_WIDTH, PIPE_HEIGHT, BACKGROUND_WIDTH, PLAYER_INDEX_GEN, DISPLAY, \
            PYG_SCREEN

    DISPLAY = display

    FPS = 30
    SCREENWIDTH = 288
    SCREENHEIGHT = 512

    SCREEN = PIL.Image.new(mode="RGB", size=(SCREENWIDTH, SCREENHEIGHT))

    IMAGES, HITMASKS = loadAssets()
    
    PLAYER_WIDTH = IMAGES['player'][0].width
    PLAYER_HEIGHT = IMAGES['player'][0].height
    PIPE_WIDTH = IMAGES['pipe'][0].width
    PIPE_HEIGHT = IMAGES['pipe'][0].height
    BACKGROUND_WIDTH = IMAGES['background'].width

    if display:
        global pygame
        import importlib
        pygame = importlib.import_module("pygame")
        pygame.init()
        FPSCLOCK = pygame.time.Clock()
        PYG_SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
        pygame.display.set_caption('Flappy Bird')
        

    PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
    BASEY = int(SCREENHEIGHT * 0.79)

    PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

def toSurface(image):
    mode = image.mode
    size = image.size
    data = image.tobytes()

    return pygame.image.fromstring(data, size, mode)

def loadAssets():
    IMAGES, HITMASKS = {}, {}

    # base (ground) sprite
    IMAGES['base'] = PIL.Image.open(BASE_PATH)

    # select random background sprites
    IMAGES['background'] = PIL.Image.open(BACKGROUND_PATH)

    # select random player sprites
    IMAGES['player'] = (
        PIL.Image.open(PLAYER_PATH[0]),
        PIL.Image.open(PLAYER_PATH[1]),
        PIL.Image.open(PLAYER_PATH[2]),
    )

    # select random pipe sprites
    IMAGES['pipe'] = (
        PIL.Image.open(PIPE_PATH).rotate(180),
        PIL.Image.open(PIPE_PATH),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    return IMAGES, HITMASKS


def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []

    for x in range(image.width):
        mask.append([])
        for y in range(image.height):
            mask[x].append(bool(image.getpixel((x, y))[3]))
    return mask


class Flappy:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].width - BACKGROUND_WIDTH

        newPipe1 = Flappy.getRandomPipe()
        newPipe2 = Flappy.getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH // 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH // 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.playerVelY = 0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10   # max vel along Y, max descend speed
        self.playerMinVelY = -8   # min vel along Y, max ascend speed
        self.playerAccY = 1   # players downward accleration
        self.playerFlapAcc = -9   # players speed on flapping
        self.playerFlapped = False  # True when player flaps

        # init screen by doing nothing
        self.frame_step([1, 0])

    def frame_step(self, input_actions):
        if DISPLAY:
            pygame.event.pump()

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        # check for score
        playerMidPos = self.playerx + PLAYER_WIDTH // 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH // 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                reward = 1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY -
                            self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = Flappy.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check if crash here
        isCrash = Flappy.checkCrash({'x': self.playerx, 'y': self.playery,
                              'index': self.playerIndex},
                             self.upperPipes, self.lowerPipes)
        if isCrash:
            terminal = True
            self.__init__()
            reward = -1

        SCREEN.paste(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.paste(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.paste(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.paste(IMAGES['base'], (self.basex, BASEY))
        SCREEN.paste(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        image_data = SCREEN

        if DISPLAY:
            # draw sprites
            PYG_SCREEN.blit(toSurface(SCREEN), (0, 0))

            pygame.display.update()
            FPSCLOCK.tick(FPS)

        return image_data, reward, terminal

    @staticmethod
    def getRandomPipe():
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(gapYs)-1)
        gapY = gapYs[index]

        gapY += int(BASEY * 0.2)
        pipeX = SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
            {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
        ]

    @staticmethod
    def checkCrash(player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = IMAGES['player'][0].width
        player['h'] = IMAGES['player'][0].height

        # if player crashes into ground
        if player['y'] + player['h'] >= BASEY - 1:
            return True
        else:

            playerRect = Rect(player['x'], player['y'],
                                    player['w'], player['h'])

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = Rect(
                    uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
                lPipeRect = Rect(
                    lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

                # player and upper/lower pipe hitmasks
                pHitMask = HITMASKS['player'][pi]
                uHitmask = HITMASKS['pipe'][0]
                lHitmask = HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = Flappy.pixelCollision(
                    playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = Flappy.pixelCollision(
                    playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return True

        return False

    @staticmethod
    def pixelCollision(rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                    return True
        return False
