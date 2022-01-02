class TD3Config:

    # (float) Std deviation of the noise used for environment exploration
    EXPLORATION_NOISE = 0.1

    # (float) Learning rate of the actor network
    POLICY_LR = 1e-3

    # (float) Learning rate of the critics network
    QNETS_LR = 1e-3

    # (float) Controls how much the target networks get updated
    TAU = 0.005

    # (float) controls SD of clipped noise applied to the calcuated next actions from the target policy network
    POLICY_NOISE = 0.2

    # (float) controls SD of clipped noise applied to the calcuated next actions from the target policy network
    POLICY_NOISE_CLIP = 0.5

    # (float) The discount value applied to the calculated target Q values
    DISCOUNT = 0.99

    # (int) How many steps are taken to initially fill the replay buffer
    WARM_START_STEPS = 10000

    # (int) The maximum number of samples that can be stored in the replay buffer at one time
    BUFFER_SIZE = 10000

    # (int) How many samples to take from the replay buffer at once
    EPISODE_LENGTH = 1000

    # (int) the number of samples in a batch
    BATCH_SIZE = 100

    # (int) The number of iterations before the policy network & target networks are updated
    POLICY_DELAY = 2

    # (str) name of OpenAI Gym environment (Note: this system is designed for a 2-dimensional pixel state space)
    GYM_ENVIRONMENT = 'CarRacing-v0'

    # (str) Name of experiement
    EXPERIMENT_NAME = 'revamp-train-1'

    # (Union[int, None]) number of GPUs to use (None -> cpu training)
    GPUS = 1

    # (int) save only the top K performing models
    SAVE_TOP_K = 1

    # (int) How often (in steps) to run validation
    VAL_CHECK_INTERVAL = 5_000

    # (str) path the .ckpt file to start training from OR load to test
    CHECKPOINT = './experiments/CarRacing-v0-TD3/training-run-5/epoch=249-reward=-0.0838297382.ckpt'

    # (int) Number of episodes ran for validation
    VAL_EPISODES = 10

    # (int) max number of steps to train for
    MAX_STEPS = 1_000_000