class TD3Config:

    # (int) Epochs
    N_EPISODES = 1

    # (int) Time steps
    T_STEPS = 1

    # (float) Std deviation of the noise used for environment exploration
    EXPLORATION_NOISE_SDEV = 1.0

    # (float) Learning rate of the actor network
    POLICY_LR = 3e-4

    # (float) Learning rate of the critics network
    QNETS_LR = 3e-4

    # (float) Controls how much the target networks get updated
    TAU = 0.6

    # (int) How many steps are taken to initially fill the replay buffer
    WARM_START_STEPS = 1000

    # (int) The maximum number of samples that can be stored in the replay buffer at one time
    BUFFER_SIZE = 1000

    # (int) How many samples to take from the replay buffer at once
    SAMPLE_SIZE = 3
