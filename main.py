"""
The main class, for training and testing agents on the environment.
"""


def get_config():
    """
    Parse command line arguments.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--experiment", type=str, help="which experiment to run")
    parser.add_argument("--np_seed", type=int, help="seed for np.random")
    parser.add_argument("--trial_lengths", nargs="+", type=int, help="trial lengths to try")

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unparsed = get_config()

    from utils import set_np_seed
    set_np_seed(args.np_seed)

    import gym
    import experiment

    # set level to ERROR so doesn't log WARN level (in particular so it doesn't WARN about automatic detecting of dtype)
    gym.logger.set_level(40)

    if args.experiment == "random_num_trials":
        experiment.random_num_trials(args.trial_lengths)
    else:
        print("Name of experiment not recognized.")
