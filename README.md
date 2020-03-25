
# States-Joeynmt
Prior work, with extracting the hidden states from LSTM joey model. And using it  on the DQN approach. 

# TODO:
* Incorporate the final step of the algorith, the update of the seq2seq model.

Command to try the DQN:
python3 -m joeynmt dqn_train reverse_model/config.yaml

Command to visualise the loss and total reward on tensorboard
tensorboard --logdir reverse_model/tensorboard_DQN/ 
