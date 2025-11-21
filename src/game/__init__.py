"""Chess game module."""
from .game import ChessGame
from bot.bot import Bot
from time import sleep

__all__ = ["ChessGame"]

results = []


def main() -> None:
    """Simple interactive chess game for testing."""
    game = ChessGame()

    bot = Bot(game, True)
    bot2 = Bot(game, False)
    
    # Print initial instructions (only once)
    game.clear_screen()
    game.print_board_state()

    white_to_move = True

    while not game.is_game_over():
        # sleep(2)
        # Clear screen and redraw board
        game.clear_screen()
        game.print_board_state()

        # move_input = input("\nEnter move: ").strip().lower()

        # if move_input == "quit":
        #     game.clear_screen()
        #     print("Game ended.")
        #     break
        # elif move_input == "undo":
        #     if game.undo_move():
        #         continue  # Redraw board
        #     else:
        #         print("No moves to undo. Press Enter to continue...")
        #         input()
        #         continue

        if(white_to_move):
            move = bot.get_move()
            white_to_move = False
        else:
            move = bot2.get_move()
            white_to_move = True


        if game.make_move(move):
            # Move successful, loop will redraw board
            continue
        else:
            # Invalid move - show error but keep board visible
            print(game.get_display_invalid_move(move_input))
            input("Press Enter to continue...")
            continue

    if game.is_game_over():
        game.clear_screen()
        game.print_game_over()


if __name__ == "__main__":
    main()

