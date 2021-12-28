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

    # (int) The number of iterations before the policy network & target networks are updated
    UPDATE_POLICY_STEPS = 10

    # (str) directory where training logs are stored
    LOG_DIR = "./training-logs"

    # (str) name of OpenAI Gym environment (Note: this system is designed for a 2-dimensional pixel state space)
    GYM_ENVIRONMENT = 'CarRacing-v0'

    # (str) Name of experiement
    RUN_NAME = 'debugging'

    # (Union[int, None]) number of GPUs to use (None -> cpu training)
    GPUS = None

    # (str) directory where saved models are stored
    SAVED_MODEL_DIR = "./saved-models"

    # (int) save only the top K performing models
    SAVE_TOP_K = 1

    # (int) validation step not implemented, however PL needs this anyway
    VAL_CHECK_INTERVAL = 2