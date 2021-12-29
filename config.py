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
    DISCOUNT = 1.0

    # (int) How many steps are taken to initially fill the replay buffer
    WARM_START_STEPS = 1000

    # (int) The maximum number of samples that can be stored in the replay buffer at one time
    BUFFER_SIZE = 1000

    # (int) How many samples to take from the replay buffer at once
    EPISODE_LENGTH = 1000

    # (int) the number of samples in a batch
    BATCH_SIZE = 32

    # (int) The number of iterations before the policy network & target networks are updated
    POLICY_DELAY = 2

    # (str) name of OpenAI Gym environment (Note: this system is designed for a 2-dimensional pixel state space)
    GYM_ENVIRONMENT = 'CarRacing-v0'

    # (str) Name of experiement
    EXPERIMENT_NAME = 'debugging'

    # (Union[int, None]) number of GPUs to use (None -> cpu training)
    GPUS = None

    # (int) save only the top K performing models
    SAVE_TOP_K = 1

    # (int) validation step not implemented, however PL needs this anyway
    VAL_CHECK_INTERVAL = 5000