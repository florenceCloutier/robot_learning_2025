import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from hw1.roble.infrastructure import pytorch_util as ptu
from hw1.roble.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        if self._discrete:
            self._logits_na = ptu.build_mlp(input_size=self._ob_dim,
                                           output_size=self._ac_dim,
                                           params=self._network)
            self._logits_na.to(ptu.device)
            self._mean_net = None
            self._logstd = None
            self._optimizer = optim.Adam(self._logits_na.parameters(),
                                        self._learning_rate)
        else:
            self._logits_na = None
            self._mean_net = ptu.build_mlp(input_size=self._ob_dim,
                                      output_size=self._ac_dim,
                                      params=self._network)
            self._mean_net.to(ptu.device)

            if self._deterministic:
                self._optimizer = optim.Adam(
                    itertools.chain(self._mean_net.parameters()),
                    self._learning_rate
                )
            else:
                self._std = nn.Parameter(
                    torch.ones(self._ac_dim, dtype=torch.float32, device=ptu.device) * 0.1
                )
                self._std.to(ptu.device)
                self._learn_policy_std = False # For now! 
                if self._learn_policy_std:
                    self._optimizer = optim.Adam(
                        itertools.chain([self._std], self._mean_net.parameters()),
                        self._learning_rate
                    )
                else:
                    self._optimizer = optim.Adam(
                        itertools.chain(self._mean_net.parameters()),
                        self._learning_rate
                    )

        if self._nn_baseline:
            self._baseline = ptu.build_mlp(
                input_size=self._ob_dim,
                output_size=1,
                params=self._network
            )
            self._baseline.to(ptu.device)
            self._baseline_optimizer = optim.Adam(
                self._baseline.parameters(),
                self._critic_learning_rate,
            )
        else:
            self._baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: 
        ## Provide the logic to produce an action from the policy
        obs_tensor = torch.FloatTensor(obs).to(ptu.device)
        with torch.no_grad():
            action_distribution = self.forward(obs_tensor)
        action = action_distribution.mean
        return action.cpu().numpy()

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self._discrete:
            logits = self._logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            device = next(self._mean_net.parameters()).device
            observation = observation.to(device)
            if self._deterministic:
                ##  TODO output for a deterministic policy
                action = self._mean_net(observation)
                return action
            else:
                ## DONE TODO output for a stochastic policy
                mean = self._mean_net(observation)
                action_distribution = distributions.Normal(mean, self._std)
        return action_distribution
    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        # pass
        raise NotImplementedError

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loss = nn.MSELoss()

    def update(
        self, observations, actions,
        adv_n=None, acs_labels_na=None, qvals=None
        ):
        obs_tensor = torch.FloatTensor(observations).to(ptu.device)
        actions_tensor = torch.FloatTensor(actions).to(ptu.device)
        
        self._optimizer.zero_grad()
        
        # Forward pass to get the predicted actions
        action_distribution = self.forward(obs_tensor)
        
        # TODO: update the policy and return the loss
        loss = self._loss(action_distribution.rsample(), actions_tensor)
        
        # Gradient update
        loss.backward()
        self._optimizer.step()
        
        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_idm(
        self, observations, actions, next_observations,
        adv_n=None, acs_labels_na=None, qvals=None
        ):
        # TODO: Transform the numpy arrays to torch tensors (for obs, next_obs and actions)
        obs_tensor = torch.FloatTensor(observations).to(ptu.device)
        next_obs_tensor = torch.FloatTensor(next_observations).to(ptu.device)
        actions_tensor = torch.FloatTensor(actions).to(ptu.device)
        
        # TODO: Create the full input to the IDM model (hint: it's not the same as the actor as it takes both obs and next_obs)
        idm_input = torch.cat([obs_tensor, next_obs_tensor], dim=-1)
        
        # TODO: Create the full input to the IDM model (hint: it's not the same as the actor as it takes both obs and next_obs)
        idm_input = torch.cat([obs_tensor, next_obs_tensor], dim=-1)
        
        # TODO: Get the predicted actions from the IDM model (hint: you need to call the forward function of the IDM model)
        predicted_actions = self.forward(idm_input)
        
        # TODO: Compute the loss using the MLP_policy loss function
        loss = self._loss(predicted_actions, actions_tensor)
        
        # TODO: Update the IDM model.
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        return {
            'Training Loss IDM': ptu.to_numpy(loss),
        }