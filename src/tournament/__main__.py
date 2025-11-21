"""Main entry point for running tournaments."""
import sys
from .tournament import Tournament


def main():
    """Run a tournament with configurable bots, games, workers, and generations."""
    print("=" * 60)
    print("Chess Bot Tournament")
    print("=" * 60)
    print()
    
    # Allow specifying number of workers and generations from command line
    num_workers = None
    num_generations = 1
    
    if len(sys.argv) > 1:
        try:
            num_workers = int(sys.argv[1])
            print(f"Using {num_workers} parallel workers (specified)")
        except ValueError:
            print(f"Invalid number of workers: {sys.argv[1]}. Using default (CPU count).")
    
    if len(sys.argv) > 2:
        try:
            num_generations = int(sys.argv[2])
            print(f"Running {num_generations} generation(s)")
        except ValueError:
            print(f"Invalid number of generations: {sys.argv[2]}. Using default (1).")
    
    tournament = Tournament(num_bots=500, games_per_bot=100, num_workers=num_workers, num_generations=num_generations)
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
    print("Top 10 bots by win rate:")
    for i, bot in enumerate(stats["top_bots"], 1):
        print(f"{i}. Bot {bot['bot_id']}: {bot['win_rate']:.2%} "
              f"({bot['wins']}W-{bot['losses']}L-{bot['draws']}D)")
    
    filepath = tournament.save_results()
    print()
    print(f"Results saved to: {filepath}")
    
    # Option to compare best bot against original bots
    # print()
    # response = input("Would you like to compare the best bot against the original bots? (y/n): ").strip().lower()
    # if response == 'y' or response == 'yes':
    #     num_games = 1
    #     try:
    #         games_input = input(f"How many games per original bot? (default: {num_games}): ").strip()
    #         if games_input:
    #             num_games = int(games_input)
    #     except ValueError:
    #         print(f"Invalid input, using default: {num_games}")
        
    #     comparison_results = tournament.compare_best_to_originals(num_games_per_original=num_games)


if __name__ == "__main__":
    main()

