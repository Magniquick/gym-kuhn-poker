from gym_kuhn_poker.envs.kuhn_poker_env import KuhnPokerEnv

# def print(message: str):
#     """Prints a message to the console."""
#     #print(message)

def play_random_game():
    env = KuhnPokerEnv(number_of_players=2, deck_size=3)
    _, _ = env.reset()
    reward_vector = None
    done = False
    print("\n=== Playing moves ===")

    while not done:
        env.render()        # prints last move + updated pots
        action = env.action_space.sample()
        obs, reward_vector, done, *_ = env.step(action)
        print(f"current observation: {obs}")
        print(f"current reward: {reward_vector}")

    env.render()
    assert reward_vector is not None, "Reward vector should not be None"
    
    print("Game over!")
    print(f"Final payouts: {reward_vector}")
    winner = reward_vector
    print(f"Winner is player {winner} (net {reward_vector} chips).")

    return winner, reward_vector, 

if __name__ == "__main__":
    play_random_game()