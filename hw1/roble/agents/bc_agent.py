from hw1.roble.infrastructure.replay_buffer import ReplayBuffer
from hw1.roble.policies.MLP_policy import MLPPolicySL
from .base_agent import BaseAgent
import torch
import numpy as np
import pickle
from hw1.roble.infrastructure import pytorch_util as ptu

class BCAgent(BaseAgent):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, env, **kwargs):
        super(BCAgent, self).__init__()

        self.env_params = kwargs
        # actor/policy
        self._actor = MLPPolicySL(
            **kwargs,
            deterministic=False,
            nn_baseline=False,

        )

        self.idm_params = kwargs
        
        # **NOT SURE** TODO: Adjust the input dimension of the IDM (hint: it's not the same as the actor as it takes both obs and next_obs)
        self.idm_params['ob_dim'] *= 2
        
        self._idm = MLPPolicySL(
            **self.idm_params,
            deterministic=True,
            nn_baseline=False,
        )

        # replay buffer
        self.reset_replay_buffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # training a BC agent refers to updating its actor using
        # the given observations and corresponding action labels
        log = self._actor.update(ob_no, ac_na)  # HW1: you will modify this
        return log

    def train_idm(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # training the IDM refers to updating the model using
        # the given observations, next_observations and corresponding action labels
        log = self._idm.update_idm(ob_no, ac_na, next_ob_no)
        return log
    
    def use_idm(self, paths):
        # we will use the IDM to label an entire dataset of expert **unlabelled** data
        self._idm.eval()
        
        all_labelled_data = []

        for episode_idx in range(len(paths)):
            observations = torch.tensor(np.array(paths[episode_idx]["observation"]), dtype=torch.float32)
            next_observations = torch.tensor(np.array(paths[episode_idx]["next_observation"]), dtype=torch.float32)

            ep_labelled_data = {
                "observation": [],
                "next_observation": [],
                "action": [],
                "terminal": np.array(paths[episode_idx]["terminal"]),
                "reward": np.array(paths[episode_idx]["reward"]),
                "image_obs" : np.array(paths[episode_idx]["image_obs"]),
            }
            
            with torch.no_grad():
                # DONE TODO: create the input to the IDM with observations and next_observations
                full_input = torch.cat([observations, next_observations], dim=1)
                # DONE TODO: query the IDM for the action (use one of the policy methods)
                action = self._idm.forward(full_input)  # Forward pass
                # action = action_distribution.sample()  # Sample action from dist


            ep_labelled_data["observation"] = observations.cpu().numpy()
            ep_labelled_data["next_observation"] = next_observations.cpu().numpy()
            ep_labelled_data["action"] = action.cpu().numpy()

            all_labelled_data.append(ep_labelled_data)
            print("Index: ", episode_idx, "was labelled")

        # Don't change: save labelled data
        save_path = self.env_params["expert_data"].replace("expert_data_", "labelled_data_")
        with open(save_path, "wb") as f:
            pickle.dump(all_labelled_data, f)
            print("Saved labelled data to labelled_data.pkl")
    
    def add_to_replay_buffer(self, paths):
        self._replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self._replay_buffer.sample_random_data(batch_size)  # HW1: you will modify this

    def save(self, path):
        return self._actor.save(path)

    def reset_replay_buffer(self):
        self._replay_buffer = ReplayBuffer(self._max_replay_buffer_size)