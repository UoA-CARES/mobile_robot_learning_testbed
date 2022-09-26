"""
Authors:  Aakaash Salvaji, Harry Taylor, David Valencia, Trevor Gee, Henry Williams
The University of Auckland

TD3 Networks
"""

from argparse import Action
import torch
import torch.nn as nn


class Critic_NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(Critic_NN, self).__init__()
        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=num_actions)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)   # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = self.h_linear_4(x)                  # No activation function here
        return x


class Actor_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor_NN, self).__init__()
        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.bn1 = nn.BatchNorm1d(hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=output_size)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.bn1(self.h_linear_3(x)))
        x = torch.tanh(self.h_linear_4(x))
        return x
# ------------------------------------------------------------------------#

# -------------------Networks for SAC ------------------------------------#


class ValueNetworkSAC(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetworkSAC, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetworkSAC(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetworkSAC, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[0])
        self.linear3 = nn.Linear(hidden_size[1], 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetworkSAC(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetworkSAC, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])

        self.mean_linear = nn.Linear(hidden_size[1], num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size[1], num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#


# -----------------Networks for Image Representation -------------------------#
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Actor_Img(nn.Module):
    def __init__(self, action_dim):
        super(Actor_Img, self).__init__()
        self.latent_dim = 256

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 5, 2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 5, 2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 5, 4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 5, 2, padding=2),  ## output size: [256, 1, 1]
            Flatten(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(30, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        return x



class Critic_Img(nn.Module):
    def __init__(self, action_dim):
        super(Critic_Img, self).__init__()
        self.latent_dim = 256

        self.encoder_critic_1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 5, 2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 5, 2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 5, 4, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),
            Flatten(),  ## output: 256
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim + action_dim, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
        )

    def forward(self, x, a):
        x1 = x
        x1 = self.encoder_critic_1(x1)
        x1 = torch.cat([x1, a], dim=1)
        x1 = self.fc1(x1)
        return x1


# -------------------Networks for Model Learning -----------------------------#

class ModelNet_probabilistic_transition(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNet_probabilistic_transition, self).__init__()
        self.mean_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size[0], bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size[0], hidden_size[1], bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size[1], hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size[2], 1)
        )
        self.std_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size[0], bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size[0], hidden_size[1], bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size[1], hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size[2], 1),
            nn.Softplus()
        )

    def forward(self, state, action):
        x   = torch.cat([state, action], dim=1)  # Concatenates the seq tensors in the given dimension
        u   = self.mean_layer(x)
        std = self.std_layer(x)
        return torch.distributions.Normal(u, std)



