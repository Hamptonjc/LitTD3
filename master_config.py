class MasterConfig:

    # (str) Name of experiement
    EXPERIMENT_NAME = 'v2-run1'

    # (str) Set the environment API type (choices: 'gym')
    ENVIRONMENT_API = 'gym'

    # (dict) settings to give to environment
    ENVIRONMENT_SETTINGS = {"id": "HalfCheetah-v4"}

    # (int) number for RNGs
    RANDOM_SEED = 314

    # (Union[int, None]) number of GPUs to use (None -> cpu training)
    GPUS = 1

    # (int) save only the top K performing models
    SAVE_TOP_K = 1

    # (int) How often (in steps) to run validation
    VAL_CHECK_INTERVAL = 5e3

    # (str) path the .ckpt file to start training from OR load to test
    CHECKPOINT = None

    # (int) Number of episodes ran for validation
    VAL_EPISODES = 10

    # (int) max number of steps to train for
    MAX_STEPS = 1e6

    # (float) Std deviation of the noise used for environment exploration
    EXPLORATION_NOISE = 0.1

    # (float) Learning rate of the actor network
    ACTOR_LR = 1e-3

    # (float) Learning rate of the critics network
    CRITICS_LR = 1e-3

    # (float) Controls how much the target networks get updated
    TAU = 0.005

    # (float) controls SD of clipped noise applied to the calcuated next actions from the target actor network
    ACTOR_NOISE = 0.2

    # (float) controls SD of clipped noise applied to the calcuated next actions from the target actor network
    ACTOR_NOISE_CLIP = 0.5

    # (float) The discount value applied to the calculated target Q values
    DISCOUNT = 0.99

    # (int) How many steps are taken to initially fill the replay buffer
    WARM_START_STEPS = 25e3

    # (int) The maximum number of samples that can be stored in the replay buffer at one time
    BUFFER_SIZE = int(1e6)

    # (int) the number of samples in a batch
    BATCH_SIZE = 256

    # (int) The number of iterations before the actor network & target actor networks are updated
    ACTOR_DELAY = 2