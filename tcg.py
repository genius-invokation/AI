from gitcg import (
    Deck, Player, Game, CreateParam, GameStatus
)

from ai import HeuristicPlayer, LLMPlayer, LMPlayer, LMTrainer

# 星柚
# AZFy20EQAUHC9UUQFVFB94QQCWJhBo8RClJxB5gRFGICCTEUDLLxi8AZDaJRDMYRDcEB
example_deck = {
    "characters": [
        1107,
        1507,
        1204
    ],
    "cards": [
        215071,
        215071,
        311206,
        311303,
        311406,
        312004,
        312004,
        312018,
        312018,
        321013,
        321013,
        322004,
        322004,
        322008,
        330001,
        332004,
        332004,
        332005,
        332006,
        332006,
        332021,
        332021,
        332022,
        332022,
        332024,
        332025,
        332032,
        332032,
        333003,
        333004
    ]
}

DECK0 = Deck(characters=example_deck["characters"], cards=example_deck["cards"])
DECK1 = Deck(characters=example_deck["characters"], cards=example_deck["cards"])

LMTRAINER = LMTrainer()

def play_game():
    game = Game(create_param=CreateParam(deck0=DECK0, deck1=DECK1))
    heuristic_id = 0
    llm_id = 1
    game.set_player(heuristic_id, HeuristicPlayer(heuristic_id))
    game.set_player(llm_id, LMPlayer(llm_id, LMTRAINER))

    game.start()
    while game.is_running():
        game.step()

    if game.status() == GameStatus.FINISHED:
        print("game over")
        print("winner is player", game.winner(), "LLMPlayer" if game.winner() == llm_id else "HeuristicPlayer" if game.winner() == heuristic_id else "draw")
        return game.winner()
    else:
        print("game aborted")
        print("game status:", game.status())
        print("error message:", game.error())
        return -1

if __name__ == "__main__":
    play_result = []
    LMTRAINER.start()
    for _ in range(20):
        play_result.append(play_game())
        if LMTRAINER.want_to_stop():
            break
    LMTRAINER.stop()
    print(play_result)
