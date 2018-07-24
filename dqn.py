import torch
import torch.nn as nn
from collections import deque
import random

import PIL.Image

class DQN(nn.Module):
    def __init__(self, input_size=(80, 80, 4), num_classes=2):
        super(DQN, self).__init__()

        # FIXME: support more input_size
        assert(input_size==(80, 80, 4)), "now only support input_size (80, 80, 4)"

        self.input_size = input_size
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_size[2], 32, kernel_size=8, padding=2, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.linear = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    def initializeWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class ReplayMem():
    def __init__(self, max_mem_size, input_size):
        self.mem = deque(maxlen=max_mem_size)
        self.input_size = input_size
        self.frame_size = input_size[0:2]
        self.time_step = input_size[2]
        self.initState()

    def initState(self):
        self.current_state = torch.zeros(self.input_size[2], *self.frame_size, dtype=torch.float)

    def getCurrentState(self):
        return self.current_state

    def getBatch(self, batch_size):
        batch = random.sample(self.mem, batch_size)
        batch = list(zip(*batch))
        batch[0] = torch.stack(batch[0])
        batch[1] = torch.stack(batch[1])
        batch[2] = torch.stack(batch[2])
        return batch

    def storeTransition(self, frame, action, reward, terminal):
        frame = self.processFrame(frame)
        state = torch.empty_like(self.current_state)
        state[0:-1] = self.current_state[1:]
        state[-1] = frame
        action = torch.tensor(action, dtype=torch.float)

        self.mem.append((self.current_state, action, state, reward, terminal))

        if not terminal:
            self.current_state = state
        else:
            self.initState()

    def processFrame(self, frame):
        frame = frame.resize(self.frame_size, PIL.Image.ANTIALIAS).convert("L")
        frame = frame.point(lambda p : 1 if p > 1 else 0)
        return torch.tensor(frame.getdata(), dtype=torch.float).view(self.frame_size)