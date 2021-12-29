class TD3Config:

    # (float) Std deviation of the noise used for environment exploration
    EXPLORATION_NOISE = 1.0

    # (float) Learning rate of the actor network
    POLICY_LR = 3e-4

    # (float) Learning rate of the critics network
    QNETS_LR = 3e-4

    # (float) Controls how much the target networks get updated
    TAU = 0.6

    # (float) controls SD of clipped noise applied to the calcuated next actions from the target policy network
    POLICY_NOISE = 1.0

    # (float) controls SD of clipped noise applied to the calcuated next actions from the target policy network
    POLICY_NOISE_CLIP = 1.0

    # (float) The discount value applied to the calculated target Q values
    DISCOUNT = 1.0

    # (int) How many steps are taken to initially fill the replay buffer
    WARM_START_STEPS = 100

    # (int) The maximum number of samples that can be stored in the replay buffer at one time
    BUFFER_SIZE = 100

    # (int) How many samples to take from the replay buffer at once
    EPISODE_LENGTH = 10

    # (int) the number of samples in a batch
    BATCH_SIZE = 16

    # (int) The number of iterations before the policy network & target networks are updated
    POLICY_DELAY = 10

    # (str) name of OpenAI Gym environment (Note: this system is designed for a 2-dimensional pixel state space)
    GYM_ENVIRONMENT = 'CarRacing-v0'

    # (str) Name of experiement
    EXPERIMENT_NAME = 'debugging'

    # (Union[int, None]) number of GPUs to use (None -> cpu training)
    GPUS = None

    # (int) save only the top K performing models
    SAVE_TOP_K = 1

    # (int) validation step not implemented, however PL needs this anyway
    VAL_CHECK_INTERVAL = 100