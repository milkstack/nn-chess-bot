"""Chess game module."""
import json
import argparse
import chess
from pathlib import Path
from .game import ChessGame
from bot.bot import Bot
from bot.neural_net import NeuralNet
from time import sleep

__all__ = ["ChessGame"]

results = []


def load_bot_by_id(bot_id: int, models_file: str = None) -> Bot:
    """
    Load a bot by ID from the saved models file.
    
    Args:
        bot_id: The ID of the bot to load
        models_file: Path to models.json file. If None, uses data/models.json
        
    Returns:
        Bot instance with loaded neural network
    """
    if models_file is None:
        # Default to data/models.json relative to project root
        project_root = Path(__file__).parent.parent.parent
        models_file = project_root / "data" / "models.json"
    else:
        models_file = Path(models_file)
    
    if not models_file.exists():
        raise FileNotFoundError(f"Models file not found: {models_file}")
    
    with open(models_file, 'r') as f:
        models_data = json.load(f)
    
    bot_id_str = str(bot_id)
    if bot_id_str not in models_data:
        available_ids = sorted([int(k) for k in models_data.keys()])
        raise ValueError(
            f"Bot ID {bot_id} not found in models file. "
            f"Available IDs: {available_ids[:20]}{'...' if len(available_ids) > 20 else ''}"
        )
    
    # Load neural network state
    bot_state = models_data[bot_id_str]
    neural_net = NeuralNet.from_state(bot_state)
    
    # Create a dummy game for bot initialization
    dummy_game = ChessGame()
    bot = Bot(dummy_game, True, quiet=True)
    bot.neural_net = neural_net
    
    return bot


def play_against_bot(bot_id: int, player_is_white: bool = True) -> None:
    """
    Play a game against a specific bot by ID.
    
    Args:
        bot_id: The ID of the bot to play against
        player_is_white: If True, player plays white; if False, player plays black
    """
    try:
        bot = load_bot_by_id(bot_id)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading bot: {e}")
        return
    
    game = ChessGame()
    bot.game = game
    bot.is_white = not player_is_white  # Bot plays opposite color of player
    
    print("=" * 60)
    print(f"Playing against Bot {bot_id}")
    print(f"You are playing as: {'White' if player_is_white else 'Black'}")
    print("=" * 60)
    print("\nEnter moves in UCI format (e.g., e2e4) or SAN format (e.g., e4)")
    print("Commands: 'undo' to undo last move, 'quit' to exit")
    print("-" * 60)
    input("\nPress Enter to start...")
    
    while not game.is_game_over():
        game.clear_screen()
        game.print_board_state()
        
        current_turn_is_white = game.get_turn()
        is_player_turn = (current_turn_is_white == player_is_white)
        
        if is_player_turn:
            # Player's turn
            move_input = input("\nEnter your move: ").strip()
            
            if move_input.lower() == "quit":
                game.clear_screen()
                print("Game ended.")
                return
            elif move_input.lower() == "undo":
                if game.undo_move():
                    continue  # Redraw board
                else:
                    print("No moves to undo. Press Enter to continue...")
                    input()
                    continue
            
            if game.make_move(move_input):
                # Move successful, loop will continue
                continue
            else:
                # Invalid move
                print(game.get_display_invalid_move(move_input))
                input("Press Enter to continue...")
                continue
        else:
            # Bot's turn
            print("\nBot is thinking...")
            move = bot.get_move()
            
            if move:
                # Convert UCI move to SAN format for display (before making the move)
                board = game.get_board()
                chess_move = chess.Move.from_uci(move)
                move_san = board.san(chess_move)
                
                if game.make_move(move):
                    # Bot move successful - print the move
                    print(f"Bot played: {move_san} ({move})")
                    input("Press Enter to continue...")
                    continue
            else:
                # Bot made invalid move or no move (shouldn't happen, but handle gracefully)
                legal_moves = game.get_legal_moves()
                if legal_moves:
                    import random
                    move = random.choice(legal_moves)
                    # Convert to SAN for display (before making the move)
                    board = game.get_board()
                    chess_move = chess.Move.from_uci(move)
                    move_san = board.san(chess_move)
                    game.make_move(move)
                    print(f"Bot played (fallback): {move_san} ({move})")
                    input("Press Enter to continue...")
                else:
                    break
    
    # Game over
    game.clear_screen()
    game.print_game_over()
    
    result = game.get_result()
    if result == "1-0":
        winner = "White" if player_is_white else "Bot"
    elif result == "0-1":
        winner = "Bot" if player_is_white else "White"
    else:
        winner = "Draw"
    
    print(f"\nWinner: {winner}")


def main() -> None:
    """Main entry point for playing chess games."""
    parser = argparse.ArgumentParser(
        description="Play chess against a bot or watch bots play",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--bot-id",
        type=int,
        help="ID of the bot to play against (from models.json)"
    )
    parser.add_argument(
        "--color",
        choices=["white", "black"],
        default="white",
        help="Color to play as (default: white)"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch two bots play against each other (requires --bot-id and optionally --bot-id-2)"
    )
    parser.add_argument(
        "--bot-id-2",
        type=int,
        help="Second bot ID for watching bot vs bot (only used with --watch)"
    )
    
    args = parser.parse_args()
    
    if args.watch:
        # Watch two bots play
        if args.bot_id is None:
            print("Error: --bot-id is required when using --watch")
            return
        
        bot_id_1 = args.bot_id
        bot_id_2 = args.bot_id_2 if args.bot_id_2 is not None else bot_id_1
        
        try:
            bot1 = load_bot_by_id(bot_id_1)
            bot2 = load_bot_by_id(bot_id_2)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading bot: {e}")
            return
        
        game = ChessGame()
        bot1.game = game
        bot1.is_white = True
        bot2.game = game
        bot2.is_white = False
        
        print("=" * 60)
        print(f"Watching Bot {bot_id_1} (White) vs Bot {bot_id_2} (Black)")
        print("=" * 60)
        input("\nPress Enter to start...")
        
        white_to_move = True
        
        while not game.is_game_over():
            game.clear_screen()
            game.print_board_state()
            
            if white_to_move:
                move = bot1.get_move()
                white_to_move = False
            else:
                move = bot2.get_move()
                white_to_move = True
            
            if game.make_move(move):
                continue
            else:
                # Invalid move - pick random legal move as fallback
                legal_moves = game.get_legal_moves()
                if legal_moves:
                    import random
                    move = random.choice(legal_moves)
                    game.make_move(move)
                else:
                    break
            
            sleep(1)  # Small delay to see moves
        
        game.clear_screen()
        game.print_game_over()
        
    elif args.bot_id is not None:
        # Play against a specific bot
        player_is_white = (args.color == "white")
        play_against_bot(args.bot_id, player_is_white)
    else:
        # Default: bot vs bot (original behavior)
        game = ChessGame()
        bot = Bot(game, True)
        bot2 = Bot(game, False)
        
        game.clear_screen()
        game.print_board_state()
        
        white_to_move = True
        
        while not game.is_game_over():
            game.clear_screen()
            game.print_board_state()
            
            if white_to_move:
                move = bot.get_move()
                white_to_move = False
            else:
                move = bot2.get_move()
                white_to_move = True
            
            if game.make_move(move):
                continue
            else:
                legal_moves = game.get_legal_moves()
                if legal_moves:
                    import random
                    move = random.choice(legal_moves)
                    game.make_move(move)
                else:
                    break
        
        if game.is_game_over():
            game.clear_screen()
            game.print_game_over()


if __name__ == "__main__":
    main()

