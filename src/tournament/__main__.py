"""Main entry point for running tournaments."""
import argparse
from .tournament import Tournament
from .config_loader import load_config


def main():
    """Run a tournament with configurable bots, games, workers, and generations."""
    # Load configuration from file first
    config = load_config()
    
    parser = argparse.ArgumentParser(
        description="Run a chess bot tournament with evolutionary generations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Values can also be configured via tournament_config.json file. "
               "Command-line arguments override config file values."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: tournament_config.json in project root)"
    )
    parser.add_argument(
        "--num-bots",
        type=int,
        default=None,
        help=f"Number of bots to generate (default from config: {config['num_bots']})"
    )
    parser.add_argument(
        "--games-per-bot",
        type=int,
        default=None,
        help=f"Number of games each bot should play (default from config: {config['games_per_bot']})"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=f"Number of parallel workers. If not specified, uses CPU count (default from config: {config['num_workers'] or 'auto'})"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=None,
        help=f"Number of evolutionary generations to run (default from config: {config['num_generations']})"
    )
    parser.add_argument(
        "--survivors-per-generation",
        type=int,
        default=None,
        help=f"Number of bots that survive each generation (default from config: {config['survivors_per_generation']})"
    )
    parser.add_argument(
        "--mutation-chance",
        type=float,
        default=None,
        help=f"Probability that a weight or bias will be mutated (0.0 to 1.0) (default from config: {config['mutation_chance']})"
    )
    parser.add_argument(
        "--mutation-amount",
        type=float,
        default=None,
        help=f"Maximum absolute value for mutation amount (default from config: {config['mutation_amount']})"
    )
    
    args = parser.parse_args()
    
    # Reload config if a custom config file was specified
    if args.config:
        config = load_config(args.config)
    
    # Use command-line arguments if provided, otherwise use config file values
    num_bots = args.num_bots if args.num_bots is not None else config["num_bots"]
    games_per_bot = args.games_per_bot if args.games_per_bot is not None else config["games_per_bot"]
    num_workers = args.num_workers if args.num_workers is not None else config["num_workers"]
    num_generations = args.num_generations if args.num_generations is not None else config["num_generations"]
    survivors_per_generation = args.survivors_per_generation if args.survivors_per_generation is not None else config["survivors_per_generation"]
    mutation_chance = args.mutation_chance if args.mutation_chance is not None else config["mutation_chance"]
    mutation_amount = args.mutation_amount if args.mutation_amount is not None else config["mutation_amount"]
    
    print("=" * 60)
    print("Chess Bot Tournament")
    print("=" * 60)
    print()
    print(f"Configuration:")
    print(f"  Number of bots: {num_bots}")
    print(f"  Games per bot: {games_per_bot}")
    if num_workers:
        print(f"  Parallel workers: {num_workers}")
    else:
        print(f"  Parallel workers: CPU count (auto)")
    print(f"  Generations: {num_generations}")
    print(f"  Survivors per generation: {survivors_per_generation}")
    print(f"  Mutation chance: {mutation_chance}")
    print(f"  Mutation amount: {mutation_amount}")
    print()
    
    tournament = Tournament(
        num_bots=num_bots,
        games_per_bot=games_per_bot,
        num_workers=num_workers,
        num_generations=num_generations,
        survivors_per_generation=survivors_per_generation,
        mutation_chance=mutation_chance,
        mutation_amount=mutation_amount
    )
    tournament.run_tournament(use_parallel=True)
    
    print()
    print("=" * 60)
    print("Tournament Statistics")
    print("=" * 60)
    
    stats = tournament.get_statistics()
    summary = stats["summary"]
    print(f"Total games played: {stats['total_games']}")
    print(f"Total wins: {summary['total_wins']}")
    print(f"Total losses: {summary['total_losses']}")
    print(f"Total draws: {summary['total_draws']}")
    print()
    print("Top 10 bots by success score:")
    for i, bot in enumerate(stats["top_bots"], 1):
        print(f"{i}. Bot {bot['bot_id']}: {bot['success']:.3f} "
              f"({bot['wins']}W-{bot['losses']}L-{bot['draws']}D)")
    
    filepath = tournament.save_results()
    print()
    print(f"Results saved to: {filepath}")
    
    # Option to compare best bot against original bots
    print()
    
    # response = input("Would you like to compare the best bot against the original bots? (y/n): ").strip().lower()
    # if response == 'y' or response == 'yes':
    if True:
        num_games = 1
        try:
            games_input = input(f"How many games per original bot? (default: {num_games}): ").strip()
            if games_input:
                num_games = int(games_input)
        except ValueError:
            print(f"Invalid input, using default: {num_games}")
        
        comparison_results = tournament.compare_best_to_originals(num_games_per_original=num_games)


if __name__ == "__main__":
    main()

