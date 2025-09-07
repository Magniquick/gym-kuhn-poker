from gym_kuhn_poker.envs.kuhn_poker_env import KuhnPokerEnv


def play_random_game():
    env = KuhnPokerEnv(number_of_players=2, deck_size=3)
    obs, _ = env.reset()
    final_rewards = None
    terminated = False
    truncated = False

    print("\n=== Playing moves ===")

    while not (terminated or truncated):
        env.render()  # prints last move + updated pots
        action = env.action_space.sample()

        obs, reward_scalar, terminated, truncated, info = env.step(action)

        print(f"Action taken: {action}")
        print(f"Current observation: {obs}")
        print(f"Scalar reward (for acting player): {reward_scalar}")
        print(f"Reward vector so far: {info['rewards']}")

        if terminated:
            final_rewards = info["rewards"]

    env.render()
    assert final_rewards is not None, "Final reward vector should not be None"

    print("Game over!")
    print(f"Final payouts: {final_rewards}")
    winner = final_rewards.index(max(final_rewards))
    print(f"Winner is player {winner} (net {final_rewards[winner]} chips).")

    return winner, final_rewards


if __name__ == "__main__":
    play_random_game()
