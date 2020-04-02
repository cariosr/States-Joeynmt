# States-Joeynmt
Prior work, with extracting the hidden states from LSTM joey model. And using it  on the DQN approach. 

The code is implemented based on the same logic used on the original **joeynmt** repository. 
On that sense, the installation is exactly de same:

<<<<<<< HEAD
1. `git clone https://github.com/joeynmt/joeynmt.git`
=======
1. `git clone <path from this repository>`
>>>>>>> 1cb3d8bd7bc7fd1822321dedd868486c913bf4ec
1. `cd joeynmt`
1. `pip3 install .`

To test it on the reverse task. Is the same procedure than the used on **joeynmt**:
Generating the synthetic data:
`python3 scripts/generate_reverse_task.py`

Create the folder to host the data:
<<<<<<< HEAD
`mkdir test/data/reverse`
`mv train* test/data/reverse/`
`mv test* test/data/reverse/`
`mv dev* test/data/reverse/`
=======
* `mkdir test/data/reverse`
* `mv train* test/data/reverse/`
* `mv test* test/data/reverse/`
* `mv dev* test/data/reverse/`
>>>>>>> 1cb3d8bd7bc7fd1822321dedd868486c913bf4ec

Modify the config file. `reverse_model/config`

An actor model is nedded. The used is the LSTM, and there is already a trained model.
But, in case is nedded to train new one. Use the train function, as follow:
<<<<<<< HEAD
`python3 -m joeynmt train configs/reverse.yaml`
And to test the performace:
`python3 -m joeynmt test reverse_model/config.yaml --output_path reverse_model/predictions`

Now, is ready to test the code related to the DQN approach. 
To run the dqn_train function, use the following command:
`python3 -m joeynmt dqn_train reverse_model/config.yaml`

To visualise the parameters. Check it with:
`tensorboard --logdir reverse_model/tensorboard_DQN/`
=======
* `python3 -m joeynmt train configs/reverse.yaml`
And to test the performace:
* `python3 -m joeynmt test reverse_model/config.yaml --output_path reverse_model/predictions`

Now, is ready to test the code related to the DQN approach. 
To run the dqn_train function, use the following command:
* `python3 -m joeynmt dqn_train reverse_model/config.yaml`

To visualise the parameters. Check it with:
* `tensorboard --logdir reverse_model/tensorboard_DQN/`
>>>>>>> 1cb3d8bd7bc7fd1822321dedd868486c913bf4ec

# TODO:
* Debug the code. Check the performace for smaller toys problems.

