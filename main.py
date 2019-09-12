"""
The main class, for running experiments.
"""


def get_config():
    """
    Parse command line arguments.
    """
    from argparse import ArgumentParser

    default_std_devs = [1, 3, 10]
    default_lrs = [0.0001, 0.001, 0.01, 0.1]

    parser = ArgumentParser()

    parser.add_argument("--experiment", type=str, required=True, help="which experiment to run")
    parser.add_argument("--num_samples", type=int, required=True, help="number of samples to use for experiment")
    parser.add_argument("--np_seed", type=int, help="seed for np.random")
    parser.add_argument("--nums_trials", nargs="+", type=int, help=("numbers of trials to try (required for "
                                                                    "{random, hill_climb, reinforce}_trials"))
    parser.add_argument("--std_devs", nargs="+", type=int, default=default_std_devs,
                        help="standard deviations to try (required for hill_climb_std_dev)")
    parser.add_argument("--lrs", nargs="+", type=int, default=default_lrs,
                        help="learning rates to try (required for reinforce_lr")

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unparsed = get_config()

    from utils import set_np_seed
    set_np_seed(args.np_seed)

    import gym
    import experiment

    # set level to ERROR so doesn't log WARN level (in particular so it doesn't WARN about automatic detecting of dtype)
    gym.logger.set_level(40)

    try:
        if args.experiment == "random_episodes":
            experiment.random_episodes(num_samples=args.num_samples)
        elif args.experiment == "hill_climb_episodes":
            experiment.hill_climb_episodes(num_samples=args.num_samples)
        elif args.experiment == "hill_climb_trials":
            experiment.hill_climb_trials(nums_trials=args.nums_trials,
                                         num_samples=args.num_samples)
        elif args.experiment == "hill_climb_std_dev":
            experiment.hill_climb_std_dev(std_devs=args.std_devs,
                                          num_samples=args.num_samples)
        elif args.experiment == "reinforce_episodes":
            experiment.reinforce_episodes(num_samples=args.num_samples)
        elif args.experiment == "reinforce_trials":
            experiment.reinforce_trials(nums_trials=args.nums_trials,
                                        num_samples=args.num_samples)
        elif args.experiment == "reinforce_lr":
            experiment.reinforce_lr(lrs=args.lrs,
                                    num_samples=args.num_samples)

    except AttributeError:
        print("One of the required arguments was not passed in.")
