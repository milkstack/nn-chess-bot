"""Tournament system for running multiple bot games."""
import random
import json
import time
import threading
import copy
import math
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, Manager

from game.game import ChessGame
from bot.bot import Bot
from bot.neural_net import NeuralNet


def _play_game_worker(white_bot_id: int, black_bot_id: int, bot_states: List[dict], 
                     progress_counter, lock, total_games: int) -> Tuple[int, int, str]:
    """
    Worker function to play a single game in a separate process.
    
    Args:
        white_bot_id: ID of bot playing white
        black_bot_id: ID of bot playing black
        bot_states: List of serialized bot neural network states
        progress_counter: Shared counter for progress tracking
        lock: Lock for thread-safe counter updates
        total_games: Total number of games (for progress calculation)
        
    Returns:
        Tuple of (white_bot_id, black_bot_id, game_result)
    """
    import random
    import chess
    
    # Reconstruct bots from serialized states
    white_net = NeuralNet.from_state(bot_states[white_bot_id])
    black_net = NeuralNet.from_state(bot_states[black_bot_id])
    
    # Create game and bots
    game = ChessGame()
    white_bot = Bot(game, True, quiet=True)
    white_bot.neural_net = white_net
    black_bot = Bot(game, False, quiet=True)
    black_bot.neural_net = black_net
    
    # Play the game
    max_moves = 200  # Prevent infinite games
    move_count = 0
    
    while not game.is_game_over() and move_count < max_moves:
        if game.get_turn():  # White's turn
            move = white_bot.get_move()
        else:  # Black's turn
            move = black_bot.get_move()
        
        if not game.make_move(move):
            # If move is invalid, pick a random legal move as fallback
            legal_moves = game.get_legal_moves()
            if legal_moves:
                move = random.choice(legal_moves)
                game.make_move(move)
            else:
                break
        
        move_count += 1
    
    result = game.get_result()
    
    # Update progress counter
    with lock:
        progress_counter.value += 1
    
    return (white_bot_id, black_bot_id, result)


class Tournament:
    """Manages a tournament between multiple chess bots."""
    
    def __init__(self, num_bots: int = 500, games_per_bot: int = 100, num_workers: int = None, 
                 num_generations: int = 1, survivors_per_generation: int = None,
                 mutation_chance: float = 0.1, mutation_amount: float = 0.5):
        """
        Initialize a tournament.
        
        Args:
            num_bots: Number of bots to generate
            games_per_bot: Number of games each bot should play
            num_workers: Number of parallel workers. If None, uses CPU count.
            num_generations: Number of evolutionary generations to run
            survivors_per_generation: Number of bots that survive each generation. 
                                     If None, defaults to num_bots // 2
            mutation_chance: Probability that a weight or bias will be mutated (0.0 to 1.0)
            mutation_amount: Maximum absolute value for mutation amount
        """
        self.num_bots = num_bots
        self.games_per_bot = games_per_bot
        self.num_workers = num_workers
        self.num_generations = num_generations
        self.survivors_per_generation = survivors_per_generation if survivors_per_generation is not None else num_bots // 2
        self.mutation_chance = mutation_chance
        self.mutation_amount = mutation_amount
        self.bots: List[Bot] = []
        self.bot_states: List[dict] = []  # Serialized bot states for multiprocessing
        self.original_bot_states: List[dict] = []  # Store original bot states for comparison
        self.results: Dict[int, Dict[str, int]] = {}  # bot_id -> {wins, losses, draws}
        self.bot_game_counts: Dict[int, int] = {}
        self.pairings = []

        
    def generate_bots(self) -> None:
        """Generate all bots for the tournament."""
        print(f"Generating {self.num_bots} bots...")
        for i in range(self.num_bots):
            # Create a dummy game just for bot initialization
            # Each bot will get its own game instance when playing
            dummy_game = ChessGame()
            bot = Bot(dummy_game, True, quiet=True)  # Color doesn't matter here, will be set per game
            self.bots.append(bot)
            # Store serialized neural network state for multiprocessing
            self.bot_states.append(bot.neural_net.get_state())
            self.results[i] = {
                "wins": 0, 
                "losses": 0, 
                "draws": 0, 
                "games_played": 0,
                "white_wins": 0,
                "black_wins": 0,
                "lineage": []  # Initial bots have no parents
            }
            self.bot_game_counts[i] = 0
        # Store original bot states for later comparison (only on first generation)
        if not self.original_bot_states:
            self.original_bot_states = copy.deepcopy(self.bot_states)
        print(f"Generated {len(self.bots)} bots.")
    
    def play_game(self, bot1: Bot, bot2: Bot, bot1_id: int, bot2_id: int) -> str:
        """
        Play a single game between two bots.
        
        Args:
            bot1: First bot (plays white)
            bot2: Second bot (plays black)
            bot1_id: ID of first bot
            bot2_id: ID of second bot
            
        Returns:
            Game result: "1-0" (white wins), "0-1" (black wins), "1/2-1/2" (draw)
        """
        game = ChessGame()
        bot1.game = game
        bot1.is_white = True
        bot2.game = game
        bot2.is_white = False
        
        max_moves = 500  # Prevent infinite games
        move_count = 0
        
        while not game.is_game_over() and move_count < max_moves:
            if game.get_turn():  # White's turn
                move = bot1.get_move()
            else:  # Black's turn
                move = bot2.get_move()
            
            if not game.make_move(move):
                # If move is invalid, pick a random legal move as fallback
                legal_moves = game.get_legal_moves()
                if legal_moves:
                    move = random.choice(legal_moves)
                    game.make_move(move)
                else:
                    break
            
            move_count += 1
        
        result = game.get_result()
        
        # Update statistics
        if result == "1-0":  # White (bot1) wins
            self.results[bot1_id]["wins"] += 1
            self.results[bot1_id]["white_wins"] += 1
            self.results[bot2_id]["losses"] += 1
        elif result == "0-1":  # Black (bot2) wins
            self.results[bot1_id]["losses"] += 1
            self.results[bot2_id]["wins"] += 1
            self.results[bot2_id]["black_wins"] += 1
        elif result == "1/2-1/2":  # Draw
            self.results[bot1_id]["draws"] += 1
            self.results[bot2_id]["draws"] += 1
        
        # Calculate games_played from wins + losses + draws to ensure accuracy
        self.results[bot1_id]["games_played"] = (
            self.results[bot1_id]["wins"] + 
            self.results[bot1_id]["losses"] + 
            self.results[bot1_id]["draws"]
        )
        self.results[bot2_id]["games_played"] = (
            self.results[bot2_id]["wins"] + 
            self.results[bot2_id]["losses"] + 
            self.results[bot2_id]["draws"]
        )
        
        return result
    
    def run_tournament(self, use_parallel: bool = True) -> None:
        """
        Run the full tournament with evolutionary generations.
        
        Args:
            use_parallel: If True, use multiprocessing for parallel execution
        """
        if not self.bots:
            self.generate_bots()
        
        print(f"Starting tournament: {self.num_bots} bots, {self.games_per_bot} games per bot, {self.num_generations} generation(s)")
        
        for generation in range(self.num_generations):
            print(f"\n{'='*60}")
            print(f"Generation {generation + 1}/{self.num_generations}")
            print(f"{'='*60}")
            
            if use_parallel:
                self._run_parallel_tournament(generation=generation)
            else:
                self._run_sequential_tournament()
            
            # Evolve population if not the last generation
            if generation < self.num_generations - 1:
                print(f"\nEvolving population for generation {generation + 2}...")
                self._evolve_population()
                print(f"Evolution complete. Starting generation {generation + 2}...")


    def _generate_round_robin_pairings(self) -> List[Tuple[int, int]]:
        """
        Generate a partial round-robin schedule where each bot plays games_per_bot games
        against unique opponents (no bot plays the same opponent twice).
        
        Returns:
            List of (white_bot_id, black_bot_id) tuples representing game matchups
        """
        # Validate that we have enough unique opponents
        max_unique_opponents = self.num_bots - 1
        if self.games_per_bot > max_unique_opponents:
            raise ValueError(
                f"Cannot generate schedule: games_per_bot ({self.games_per_bot}) > "
                f"available unique opponents ({max_unique_opponents}). "
                f"Each bot can only play against {max_unique_opponents} other bots."
            )
        
        # Reset pairings and game counts
        self.pairings = []
        self.bot_game_counts = {i: 0 for i in range(self.num_bots)}
        
        # Track which opponents each bot has already been matched with
        # opponent_sets[bot_id] = set of bot_ids this bot has already played
        opponent_sets = {i: set() for i in range(self.num_bots)}
        
        for bot_id in range(self.num_bots):
            num_games_to_play = self.games_per_bot - self.bot_game_counts[bot_id]
            for i in range(num_games_to_play):
                # Get bots that haven't reached their game limit
                bots_with_free_pairings = {
                    k: v for k, v in self.bot_game_counts.items() 
                    if v < self.games_per_bot
                }

                
                # Filter to only include opponents this bot hasn't played yet (and not itself)
                available_opponents = [
                    opp_id for opp_id in bots_with_free_pairings.keys()
                    if opp_id != bot_id and opp_id not in opponent_sets[bot_id]
                ]
                
                # If no available opponents, skip (shouldn't happen with proper validation)
                if not available_opponents:
                    break
                
                # Randomly select an opponent
                selected_bot_id = random.choice(available_opponents)
                
                # Randomly assign colors
                if random.random() < 0.5:
                    white_bot_id = bot_id
                    black_bot_id = selected_bot_id
                else:
                    white_bot_id = selected_bot_id
                    black_bot_id = bot_id
                
                self.pairings.append((white_bot_id, black_bot_id))
                self.bot_game_counts[bot_id] += 1
                self.bot_game_counts[selected_bot_id] += 1
                
                # Mark that these bots have played each other
                opponent_sets[bot_id].add(selected_bot_id)
                opponent_sets[selected_bot_id].add(bot_id)
        
        # Verify that each bot got the expected number of games
        missing_games = []
        for bot_id in range(self.num_bots):
            count = self.bot_game_counts.get(bot_id, 0)
            if count < self.games_per_bot:
                missing_games.append((bot_id, count))
        
        if missing_games:
            print(f"Warning: Some bots did not get the full {self.games_per_bot} games:")
            for bot_id, count in missing_games[:10]:  # Show first 10
                print(f"  Bot {bot_id}: {count} games")
            if len(missing_games) > 10:
                print(f"  ... and {len(missing_games) - 10} more")

        return self.pairings
    
    def _run_sequential_tournament(self) -> None:
        """Run tournament sequentially (original implementation)."""
        # Generate round-robin schedule
        game_tasks = self._generate_round_robin_pairings()

        print(f"Generated {len(game_tasks)} game tasks (partial round-robin)")

        return
        
        total_games = len(game_tasks)
        games_played = 0
        start_time = time.time()
        
        # Play each game in the schedule
        for white_bot_id, black_bot_id in game_tasks:
            white_bot = self.bots[white_bot_id]
            black_bot = self.bots[black_bot_id]
            
            result = self.play_game(white_bot, black_bot, white_bot_id, black_bot_id)
            games_played += 1
            
            # Continuous progress updates
            if games_played % 10 == 0 or games_played == total_games:
                elapsed = time.time() - start_time
                rate = games_played / elapsed if elapsed > 0 else 0
                remaining = (total_games - games_played) / rate if rate > 0 else 0
                percentage = (games_played * 100) // total_games
                print(f"\rProgress: {games_played}/{total_games} games ({percentage}%) | "
                      f"Rate: {rate:.1f} games/sec | "
                      f"ETA: {remaining:.0f}s", end='', flush=True)
        
        print(f"\rProgress: {total_games}/{total_games} games (100%) | Complete!{' ' * 50}")
        print(f"\nTournament complete! {games_played} games played.")
    
    def _run_parallel_tournament(self, generation: int) -> None:
        """Run tournament using multiprocessing."""
        import multiprocessing as mp
        
        if self.num_workers is None:
            self.num_workers = mp.cpu_count()
        
        print(f"Using {self.num_workers} parallel workers")
        
        # Generate partial round-robin schedule
        # Each bot plays games_per_bot games against unique opponents
        game_tasks = self._generate_round_robin_pairings()
        
        total_games = len(game_tasks)
        print(f"Generated {total_games} game tasks (partial round-robin)")
        
        # Create a shared counter for progress tracking
        manager = Manager()
        progress_counter = manager.Value('i', 0)
        lock = manager.Lock()
        stop_progress = threading.Event()
        
        # Start progress monitoring thread
        def progress_monitor():
            """Monitor and print progress updates."""
            last_count = 0
            start_time = time.time()
            while not stop_progress.is_set():
                with lock:
                    current_count = progress_counter.value
                
                if current_count != last_count:
                    elapsed = time.time() - start_time
                    if current_count > 0:
                        rate = current_count / elapsed if elapsed > 0 else 0
                        remaining = (total_games - current_count) / rate if rate > 0 else 0
                        percentage = (current_count * 100) // total_games

                        total_time_remaining_hours = (total_games * (self.num_generations - generation) / rate) / 3600 if rate > 0 else float('inf')
                        # Print progress with rate and ETA
                        print(f"\rProgress: {current_count}/{total_games} games ({percentage}%) | "
                              f"Rate: {rate:.1f} games/sec | "
                              f"ETA: {remaining:.0f}s | " 
                              f"TOTAL ETA REMAINING: {total_time_remaining_hours:.2f} hours", end='', flush=True)

                        
                        
                        last_count = current_count
                
                if current_count >= total_games:
                    break
                
                time.sleep(0.5)  # Update every 0.5 seconds
        
        progress_thread = threading.Thread(target=progress_monitor, daemon=True)
        progress_thread.start()
        
        # Create tasks with bot_states included (each task is a tuple that worker can unpack)
        tasks_with_states = [
            (white_id, black_id, self.bot_states, progress_counter, lock, total_games)
            for white_id, black_id in game_tasks
        ]
        
        # Play games in parallel
        try:
            with Pool(processes=self.num_workers) as pool:
                results = pool.starmap(_play_game_worker, tasks_with_states)
        finally:
            # Stop progress monitoring
            stop_progress.set()
            progress_thread.join(timeout=1)
            # Print final progress line
            print(f"\rProgress: {total_games}/{total_games} games (100%) | Complete!{' ' * 50}")
        
        # Aggregate results
        for white_id, black_id, result in results:
            if result == "1-0":  # White wins
                self.results[white_id]["wins"] += 1
                self.results[white_id]["white_wins"] += 1
                self.results[black_id]["losses"] += 1
            elif result == "0-1":  # Black wins
                self.results[white_id]["losses"] += 1
                self.results[black_id]["wins"] += 1
                self.results[black_id]["black_wins"] += 1
            elif result == "1/2-1/2":  # Draw
                self.results[white_id]["draws"] += 1
                self.results[black_id]["draws"] += 1
            
            # Calculate games_played from wins + losses + draws to ensure accuracy
            self.results[white_id]["games_played"] = (
                self.results[white_id]["wins"] + 
                self.results[white_id]["losses"] + 
                self.results[white_id]["draws"]
            )
            self.results[black_id]["games_played"] = (
                self.results[black_id]["wins"] + 
                self.results[black_id]["losses"] + 
                self.results[black_id]["draws"]
            )
        
        print(f"\nTournament complete! {total_games} games played.")
    
    def get_statistics(self) -> Dict:
        """Get tournament statistics."""
        # Create a copy of results without neural network states
        bot_results = {}
        for bot_id, result in self.results.items():
            # Create a copy of the result dictionary (without neural_net)
            bot_result = result.copy()
            bot_results[bot_id] = bot_result
        
        stats = {
            "total_bots": self.num_bots,
            "games_per_bot": self.games_per_bot,
            "total_games": self.num_bots * self.games_per_bot,
            "bot_results": bot_results,
            "summary": {
                "total_wins": sum(r["wins"] for r in self.results.values()),
                "total_losses": sum(r["losses"] for r in self.results.values()),
                "total_draws": sum(r["draws"] for r in self.results.values()),
            }
        }
        
        # Calculate success scores (wins - losses) / games_played
        # Ensure games_played is accurate (wins + losses + draws)
        # Also ensure white_wins and black_wins exist (for backwards compatibility)
        success_scores = []
        for bot_id, result in self.results.items():
            # Recalculate games_played to ensure accuracy
            result["games_played"] = result["wins"] + result["losses"] + result["draws"]
            # Initialize white_wins and black_wins if they don't exist (for old data)
            if "white_wins" not in result:
                result["white_wins"] = 0
            if "black_wins" not in result:
                result["black_wins"] = 0
            # Initialize lineage if it doesn't exist (for old data)
            if "lineage" not in result:
                result["lineage"] = []
            total = result["games_played"]
            if total > 0:
                # success = (wins - losses) / games_played
                success = (result["wins"] - result["losses"]) / total
                success_scores.append((bot_id, success, result))
        
        success_scores.sort(key=lambda x: x[1], reverse=True)
        stats["top_bots"] = [
            {
                "bot_id": bot_id,
                "success": success,
                "wins": result["wins"],
                "losses": result["losses"],
                "draws": result["draws"],
                "games_played": result["games_played"],
                "white_wins": result.get("white_wins", 0),
                "black_wins": result.get("black_wins", 0),
                "lineage": result.get("lineage", [])
            }
            for bot_id, success, result in success_scores[:10]
        ]
        
        return stats
    
    def save_results(self, filepath: str = None) -> str:
        """
        Save tournament results to a JSON file.
        Also saves neural network models to a separate models.json file.
        
        Args:
            filepath: Optional path to save file. If None, uses data/tournament_results.json
            
        Returns:
            Path to the saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_dir = Path(__file__).parent.parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            filepath = str(data_dir / f"tournament_results.json")
        
        stats = self.get_statistics()
        stats["timestamp"] = datetime.now().isoformat()
        
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)
        
        # Save neural network models to a separate file
        data_dir = Path(filepath).parent
        models_filepath = str(data_dir / "models.json")
        models_data = {}
        for bot_id in range(len(self.bot_states)):
            models_data[str(bot_id)] = self.bot_states[bot_id]
        
        with open(models_filepath, "w") as f:
            json.dump(models_data, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        print(f"Models saved to: {models_filepath}")
        return filepath
    
    def _evolve_population(self) -> None:
        """
        Evolve the population by selecting the top survivors based on success score
        and creating mutated children from each parent.
        
        IMPROVEMENT: Added elitism - always keep the best bot unchanged to prevent
        losing the best solution found so far.
        """
        # Calculate success scores for all bots: (wins - losses) / games_played
        success_scores = []
        for bot_id, result in self.results.items():
            total = result["games_played"]
            if total > 0:
                success = (result["wins"] - result["losses"]) / total
                success_scores.append((bot_id, success))
            else:
                # If a bot hasn't played any games, give it 0 success score
                success_scores.append((bot_id, 0.0))
        
        # Sort by success score (descending)
        success_scores.sort(key=lambda x: x[1], reverse=True)
        
        # ELITISM: Always keep the best bot unchanged (no mutation)
        best_bot_id = success_scores[0][0]
        best_bot_state = self.bot_states[best_bot_id]
        best_bot_result = self.results[best_bot_id].copy()
        best_bot_result["games_played"] = 0  # Reset for new generation
        best_bot_result["wins"] = 0
        best_bot_result["losses"] = 0
        best_bot_result["draws"] = 0
        best_bot_result["white_wins"] = 0
        best_bot_result["black_wins"] = 0
        
        # Select top survivors (excluding best, which we'll add separately)
        top_survivor_ids = [bot_id for bot_id, _ in success_scores[:self.survivors_per_generation]]
        
        print(f"Selected top {self.survivors_per_generation} bots (success scores: {[f'{s:.3f}' for _, s in success_scores[:self.survivors_per_generation]]})")
        print(f"Elitism: Keeping best bot (ID: {best_bot_id}, score: {success_scores[0][1]:.3f}) unchanged")
        
        # Create new bots from top survivors as parents
        new_bots = []
        new_bot_states = []
        new_results = {}
        new_bot_game_counts = {}
        
        # Add elite bot first (unchanged)
        best_bot_net = NeuralNet.from_state(best_bot_state)
        dummy_game = ChessGame()
        elite_bot = Bot(dummy_game, True, quiet=True)
        elite_bot.neural_net = best_bot_net
        new_bots.append(elite_bot)
        new_bot_states.append(best_bot_state)
        new_results[0] = best_bot_result
        new_bot_game_counts[0] = 0
        
        # Calculate children per parent for remaining slots (num_bots - 1 for elite)
        remaining_slots = self.num_bots - 1
        children_per_parent = math.ceil(remaining_slots / len(top_survivor_ids))
        
        for parent_id in top_survivor_ids:
            # Get parent neural network
            parent_net = NeuralNet.from_state(self.bot_states[parent_id])
            
            # Create children from this parent
            for child_num in range(children_per_parent):
                if len(new_bots) >= self.num_bots:
                    break
                    
                # Create child with parent (this will automatically mutate)
                child_net = NeuralNet(parent_net.layer_config, parent=parent_net,
                                    mutation_chance=self.mutation_chance, 
                                    mutation_amount=self.mutation_amount)
                
                # Create a dummy game for bot initialization
                dummy_game = ChessGame()
                child_bot = Bot(dummy_game, True, quiet=True)
                child_bot.neural_net = child_net
                
                new_bots.append(child_bot)
                new_bot_states.append(child_net.get_state())
                
                # Initialize results for new bot
                new_bot_id = len(new_bots) - 1
                # Get parent's lineage and append parent's ID
                parent_lineage = self.results[parent_id].get("lineage", []).copy()
                child_lineage = parent_lineage + [parent_id]
                
                new_results[new_bot_id] = {
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "games_played": 0,
                    "white_wins": 0,
                    "black_wins": 0,
                    "lineage": child_lineage
                }
                new_bot_game_counts[new_bot_id] = 0
        
        # Replace old population with new one
        self.bots = new_bots
        self.bot_states = new_bot_states
        self.results = new_results
        self.bot_game_counts = new_bot_game_counts
        
        print(f"Created {len(new_bots)} new bots (1 elite + {len(new_bots) - 1} children from {len(top_survivor_ids)} parents)")
    
    def compare_best_to_originals(self, num_games_per_original: int = 1) -> Dict:
        """
        Play games between the best bot and the original set of bots.
        
        Args:
            num_games_per_original: Number of games to play against each original bot
            
        Returns:
            Dictionary with comparison results
        """
        if not self.original_bot_states:
            print("No original bot states stored. Cannot compare.")
            return {}
        
        # Find the best bot
        success_scores = []
        for bot_id, result in self.results.items():
            total = result["games_played"]
            if total > 0:
                success = (result["wins"] - result["losses"]) / total
            else:
                success = 0.0
            success_scores.append((bot_id, success))
        
        success_scores.sort(key=lambda x: x[1], reverse=True)
        best_bot_id = success_scores[0][0]
        best_bot_state = self.bot_states[best_bot_id]
        best_bot_net = NeuralNet.from_state(best_bot_state)
        
        print(f"\n{'='*60}")
        print(f"Comparing Best Bot (ID: {best_bot_id}, Success Score: {success_scores[0][1]:.3f})")
        print(f"Against {len(self.original_bot_states)} Original Bots")
        print(f"{'='*60}")
        print(f"Playing {num_games_per_original} game(s) against each original bot...")
        
        comparison_results = {
            "best_bot_id": best_bot_id,
            "best_bot_success": success_scores[0][1],
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "vs_original_results": []
        }
        
        total_games = len(self.original_bot_states) * num_games_per_original
        games_played = 0
        
        for orig_idx, original_state in enumerate(self.original_bot_states):
            original_net = NeuralNet.from_state(original_state)
            orig_wins = 0
            orig_losses = 0
            orig_draws = 0
            
            for game_num in range(num_games_per_original):
                # Alternate colors
                best_plays_white = (game_num % 2 == 0)
                
                game = ChessGame()
                best_bot = Bot(game, best_plays_white, quiet=True)
                best_bot.neural_net = best_bot_net
                
                original_bot = Bot(game, not best_plays_white, quiet=True)
                original_bot.neural_net = original_net
                
                # Play the game
                max_moves = 200
                move_count = 0
                
                while not game.is_game_over() and move_count < max_moves:
                    if game.get_turn():  # White's turn
                        move = best_bot.get_move() if best_plays_white else original_bot.get_move()
                    else:  # Black's turn
                        move = original_bot.get_move() if best_plays_white else best_bot.get_move()
                    
                    if not game.make_move(move):
                        legal_moves = game.get_legal_moves()
                        if legal_moves:
                            move = random.choice(legal_moves)
                            game.make_move(move)
                        else:
                            break
                    
                    move_count += 1
                
                result = game.get_result()
                games_played += 1
                comparison_results["games_played"] += 1
                
                if best_plays_white:
                    if result == "1-0":  # Best bot (white) wins
                        comparison_results["wins"] += 1
                        orig_losses += 1
                    elif result == "0-1":  # Original bot (black) wins
                        comparison_results["losses"] += 1
                        orig_wins += 1
                    elif result == "1/2-1/2":  # Draw
                        comparison_results["draws"] += 1
                        orig_draws += 1
                else:
                    if result == "0-1":  # Best bot (black) wins
                        comparison_results["wins"] += 1
                        orig_losses += 1
                    elif result == "1-0":  # Original bot (white) wins
                        comparison_results["losses"] += 1
                        orig_wins += 1
                    elif result == "1/2-1/2":  # Draw
                        comparison_results["draws"] += 1
                        orig_draws += 1
                
                # Progress update
                if games_played % 10 == 0 or games_played == total_games:
                    percentage = (games_played * 100) // total_games
                    print(f"\rProgress: {games_played}/{total_games} games ({percentage}%)", end='', flush=True)
            
            # Store results for this original bot
            comparison_results["vs_original_results"].append({
                "original_bot_id": orig_idx,
                "best_bot_wins": num_games_per_original - orig_wins - orig_draws,
                "original_bot_wins": orig_wins,
                "draws": orig_draws
            })
        
        print(f"\rProgress: {total_games}/{total_games} games (100%) | Complete!{' ' * 20}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("Comparison Results:")
        print(f"{'='*60}")
        print(f"Best Bot (ID: {best_bot_id}) Performance:")
        print(f"  Wins: {comparison_results['wins']}")
        print(f"  Losses: {comparison_results['losses']}")
        print(f"  Draws: {comparison_results['draws']}")
        if comparison_results["games_played"] > 0:
            success = (comparison_results["wins"] - comparison_results["losses"]) / comparison_results["games_played"]
            print(f"  Success Score: {success:.3f}")
        
        return comparison_results

