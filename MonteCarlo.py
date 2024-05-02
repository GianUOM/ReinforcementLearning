import random
import numpy as np
import matplotlib.pyplot as plt

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

class Deck:
    def __init__(self):
        self.cards = []
        self.reset()

    def reset(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['Spades', 'Clubs', 'Hearts', 'Diamonds']
        self.cards = [Card(rank, suit) for rank in ranks for suit in suits]
        random.shuffle(self.cards)

    def draw_card(self):
        return self.cards.pop()

class BlackjackRound:
    def __init__(self):
        self.deck = Deck()
        self.player_cards = []
        self.dealer_cards = []

    def start(self):
        self.deck.reset()
        self.player_cards = [self.deck.draw_card(), self.deck.draw_card()]
        self.dealer_cards = [self.deck.draw_card()]

    def hit(self):
        self.player_cards.append(self.deck.draw_card())

    def stand(self):
        while self.get_sum(self.dealer_cards) < 17:
            self.dealer_cards.append(self.deck.draw_card())

    def get_sum(self, cards):
        sum = 0
        has_ace = False
        for card in cards:
            if card.rank in ['J', 'Q', 'K']:
                sum += 10
            elif card.rank == 'A':
                sum += 11
                has_ace = True
            else:
                sum += int(card.rank)
        if sum > 21 and has_ace:
            sum -= 10
        return sum

    def get_outcome(self):
        player_sum = self.get_sum(self.player_cards)
        dealer_sum = self.get_sum(self.dealer_cards)
        if player_sum > 21:
            return 'Loss'
        elif dealer_sum > 21:
            return 'Win'
        elif player_sum > dealer_sum:
            return 'Win'
        elif player_sum < dealer_sum:
            return 'Loss'
        else:
            return 'Draw'

class RLAgent:
    def __init__(self):
        self.q_values = {}
        self.action_counts = {}

    def choose_action(self, state, epsilon):
        if state not in self.q_values or not self.q_values[state]:
            return np.random.choice(['HIT', 'STAND'])

        if random.random() < epsilon:
            return np.random.choice(['HIT', 'STAND'])
        else:
            return max(self.q_values[state], key=self.q_values[state].get)

    def generate_agent_state(self, player_sum, dealer_card, has_ace):
        return (player_sum, dealer_card, has_ace)

    def update_action_counts(self, state, action):
        if state not in self.action_counts:
            self.action_counts[state] = {}
        if action not in self.action_counts[state]:
            self.action_counts[state][action] = 0
        self.action_counts[state][action] += 1

def monte_carlo_exploring_starts(num_episodes, alpha):
    agent = RLAgent()
    for episode in range(num_episodes):
        round = BlackjackRound()
        round.start()
        state = agent.generate_agent_state(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
        episode_actions = []
        episode_rewards = []

        # Exploring Starts
        if round.get_sum(round.player_cards) in range(12, 21):
            action = np.random.choice(['HIT', 'STAND'])
            episode_actions.append(action)
        else:
            action = agent.choose_action(state, 1)  # Full exploration
            episode_actions.append(action)

        while True:
            if action == 'HIT':
                round.hit()
                if round.get_sum(round.player_cards) > 21:
                    episode_rewards.append(-1)
                    break
            else:
                round.stand()
                dealer_sum = round.get_sum(round.dealer_cards)
                player_sum = round.get_sum(round.player_cards)
                if dealer_sum > 21:
                    episode_rewards.append(1)
                    break
                elif player_sum > dealer_sum:
                    episode_rewards.append(1)
                    break
                elif player_sum < dealer_sum:
                    episode_rewards.append(-1)
                    break
                else:
                    episode_rewards.append(0)
                    break

            next_state = agent.generate_agent_state(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
            state = next_state

            # Exploring Starts
            if round.get_sum(round.player_cards) in range(12, 21):
                action = np.random.choice(['HIT', 'STAND'])
                episode_actions.append(action)
            else:
                action = agent.choose_action(state, 1)  # Full exploration
                episode_actions.append(action)

        for i in range(len(episode_actions)):
            state_action = (state, episode_actions[i])
            G = sum(episode_rewards[i:])
            # No Q-values update in exploring starts
            pass

    return agent.q_values

def monte_carlo_non_exploring_starts(num_episodes, alpha, epsilon_func):
    agent = RLAgent()
    for episode in range(num_episodes):
        round = BlackjackRound()
        round.start()
        state = agent.generate_agent_state(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
        episode_actions = []
        episode_rewards = []

        action = agent.choose_action(state, epsilon_func(episode+1))  # Use epsilon function
        episode_actions.append(action)

        while True:
            if action == 'HIT':
                round.hit()
                if round.get_sum(round.player_cards) > 21:
                    episode_rewards.append(-1)
                    break
            else:
                round.stand()
                dealer_sum = round.get_sum(round.dealer_cards)
                player_sum = round.get_sum(round.player_cards)
                if dealer_sum > 21:
                    episode_rewards.append(1)
                    break
                elif player_sum > dealer_sum:
                    episode_rewards.append(1)
                    break
                elif player_sum < dealer_sum:
                    episode_rewards.append(-1)
                    break
                else:
                    episode_rewards.append(0)
                    break

            next_state = agent.generate_agent_state(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
            state = next_state

            action = agent.choose_action(state, epsilon_func(episode+1))  # Use epsilon function
            episode_actions.append(action)

        for i in range(len(episode_actions)):
            state_action = (state, episode_actions[i])
            G = sum(episode_rewards[i:])
            # No Q-values update in non-exploring starts
            pass

    return agent.q_values

def run_episodes_and_extract_info(agent_function, num_episodes, alpha, epsilon_func):
    agent = RLAgent()
    wins = 0
    losses = 0
    draws = 0
    episode_results = {'Win': 0, 'Loss': 0, 'Draw': 0}
    unique_state_action_pairs = set()  # Initialize a set to store unique state-action pairs
    action_counts = {}  # Initialize a dictionary to store action counts
    q_values = {}  # Initialize a dictionary to store Q values
    wins_per_episode = []

    for episode in range(1, num_episodes + 1):
        round = BlackjackRound()
        round.start()
        state = agent.generate_agent_state(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
        episode_actions = []
        episode_rewards = []

        action = agent.choose_action(state, epsilon_func(episode))  # Use epsilon function
        episode_actions.append(action)

        while True:
            if action == 'HIT':
                round.hit()
                if round.get_sum(round.player_cards) > 21:
                    episode_rewards.append(-1)
                    break
            else:
                round.stand()
                dealer_sum = round.get_sum(round.dealer_cards)
                player_sum = round.get_sum(round.player_cards)
                if dealer_sum > 21:
                    episode_rewards.append(1)
                    break
                elif player_sum > dealer_sum:
                    episode_rewards.append(1)
                    break
                elif player_sum < dealer_sum:
                    episode_rewards.append(-1)
                    break
                else:
                    episode_rewards.append(0)
                    break

            next_state = agent.generate_agent_state(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
            state = next_state

            action = agent.choose_action(state, epsilon_func(episode))  # Use epsilon function
            episode_actions.append(action)

            if episode_rewards and episode_rewards[-1] == 1:
                wins_per_episode.append(1)
            else:
                wins_per_episode.append(0)

            # Store unique state-action pairs
            unique_state_action_pairs.add((state, action))

            # Update action counts
            state_action_pair = (state, action)
            if state_action_pair in action_counts:
                action_counts[state_action_pair] += 1
            else:
                action_counts[state_action_pair] = 1

            # Update Q values
            if state not in q_values:
                q_values[state] = {}
            if action not in q_values[state]:
                q_values[state][action] = 0

            # Update Q value using sample-based update rule
            q_values[state][action] += alpha * (sum(episode_rewards) - q_values[state][action])

        episode_result = round.get_outcome()
        episode_results[episode_result] += 1

        if episode % 1000 == 0:
            wins += episode_results['Win']
            losses += episode_results['Loss']
            draws += episode_results['Draw']
            print(f"Episodes {episode-999}-{episode}: Wins - {episode_results['Win']}, Losses - {episode_results['Loss']}, Draws - {episode_results['Draw']}")
            episode_results = {'Win': 0, 'Loss': 0, 'Draw': 0}


    return wins_per_episode, wins, losses, draws, len(unique_state_action_pairs), action_counts, q_values

# Configuration 1: Exploring Starts
num_episodes = 100000
alpha = 0.1
wins_per_episode, wins_explore, losses_explore, draws_explore, unique_pairs_explore, action_counts_explore, q_values_explore= run_episodes_and_extract_info(monte_carlo_exploring_starts, num_episodes, alpha, lambda k: 1 / k)

# Configuration 2: Non-exploring Starts with 𝜖 = 1/k
num_episodes = 100000
alpha = 0.1
wins_per_episode1, wins_non_explore_1, losses_non_explore_1, draws_non_explore_1, unique_pairs_non_explore_1, action_counts_non_explore_1, q_values_non_explore_1,  = run_episodes_and_extract_info(monte_carlo_non_exploring_starts, num_episodes, alpha, lambda k: 1 / k)

# Configuration 3: Non-exploring Starts with 𝜖 = e^(-k/1000)
num_episodes = 100000
alpha = 0.1
wins_per_episode2, wins_non_explore_2, losses_non_explore_2, draws_non_explore_2, unique_pairs_non_explore_2, action_counts_non_explore_2, q_values_non_explore_2 = run_episodes_and_extract_info(monte_carlo_non_exploring_starts, num_episodes, alpha, lambda k: np.exp(-k / 1000))

# Configuration 4: Non-exploring Starts with 𝜖 = e^(-k/10000)
num_episodes = 100000
alpha = 0.1
wins_per_episode3, wins_non_explore_3, losses_non_explore_3, draws_non_explore_3, unique_pairs_non_explore_3, action_counts_non_explore_3, q_values_non_explore_3,  = run_episodes_and_extract_info(monte_carlo_non_exploring_starts, num_episodes, alpha, lambda k: np.exp(-k / 10000))

print('-----------------------------------------------------------------------')
print("\nResults:\n")
print("Exploring Starts:")
print(f"Wins: {wins_explore}, Losses: {losses_explore}, Draws: {draws_explore}")
print(f"Unique state-action pairs explored: {unique_pairs_explore}")
print(f"Action counts for exploring starts: {action_counts_explore}")
print(f"Estimated Q-Values for each unique state-action pair: {q_values_explore}\n")

print("Non-exploring Starts with 𝜖 = 1/k:")
print(f"Wins: {wins_non_explore_1}, Losses: {losses_non_explore_1}, Draws: {draws_non_explore_1}")
print(f"Unique state-action pairs explored: {unique_pairs_non_explore_1}")
print(f"Action counts for non-exploring starts with 𝜖 = 1/k: {action_counts_non_explore_1}")
print(f"Estimated Q-Values for each unique state-action pair: {q_values_explore}\n")

print("Non-exploring Starts with 𝜖 = e^(-k/1000):")
print(f"Wins: {wins_non_explore_2}, Losses: {losses_non_explore_2}, Draws: {draws_non_explore_2}")
print(f"Unique state-action pairs explored: {unique_pairs_non_explore_2}")
print(f"Action counts for non-exploring starts with 𝜖 = e^(-k/1000): {action_counts_non_explore_2}")
print(f"Estimated Q-Values for each unique state-action pair: {q_values_explore}\n")

print("Non-exploring Starts with 𝜖 = e^(-k/10000):")
print(f"Wins: {wins_non_explore_3}, Losses: {losses_non_explore_3}, Draws: {draws_non_explore_3}")
print(f"Unique state-action pairs explored: {unique_pairs_non_explore_3}")
print(f"Action counts for non-exploring starts with 𝜖 = e^(-k/10000): {action_counts_non_explore_3}")
print(f"Estimated Q-Values for each unique state-action pair: {q_values_explore}\n")



# Plot the win count over episodes
# totalEpisodes = range(1, num_episodes + 1)
# plt.plot(totalEpisodes, wins_per_episode, label='Exploring Starts')
# plt.plot(totalEpisodes, wins_per_episode1, label='Non-exploring Starts (𝜖 = 1/k)')
# plt.plot(totalEpisodes, wins_per_episode2, label='Non-exploring Starts (𝜖 = e^(-k/1000))')
# plt.plot(totalEpisodes, wins_per_episode3, label='Non-exploring Starts (𝜖 = e^(-k/10000))')
# plt.yscale('log')
# plt.xlabel('Number of Episodes')
# plt.ylabel('Win Count')
# plt.title('Win Count Over Episodes')
# plt.legend()
# plt.show()

# def plot_state_action_counts(action_counts, title, top_n=10):
#     sorted_counts = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
#     state_action_pairs, counts = zip(*sorted_counts)
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(len(state_action_pairs)), counts, color='skyblue', alpha=0.8)
#     plt.xticks(range(len(state_action_pairs)), state_action_pairs, rotation=90, fontsize=10)
#     plt.ylabel('Count', fontsize=12)
#     plt.xlabel('State-Action Pair', fontsize=12)
#     plt.title(title, fontsize=14)
#     plt.grid(axis='y', linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.show()

# # Configuration 1: Exploring Starts
# plot_state_action_counts(action_counts_explore, "State-Action Counts for Exploring Starts")

# # Configuration 2: Non-exploring Starts with 𝜖 = 1/k
# plot_state_action_counts(action_counts_non_explore_1, "State-Action Counts for Non-Exploring Starts (𝜖 = 1/k)")

# # Configuration 3: Non-exploring Starts with 𝜖 = e^(-k/1000)
# plot_state_action_counts(action_counts_non_explore_2, "State-Action Counts for Non-Exploring Starts (𝜖 = e^(-k/1000))")

# # Configuration 4: Non-exploring Starts with 𝜖 = e^(-k/10000)
# plot_state_action_counts(action_counts_non_explore_3, "State-Action Counts for Non-Exploring Starts (𝜖 = e^(-k/10000))")

# Aggregate unique state-action pairs counts for each algorithm
# unique_pairs_explore_all = unique_pairs_explore
# unique_pairs_non_explore_1_all = unique_pairs_non_explore_1
# unique_pairs_non_explore_2_all = unique_pairs_non_explore_2
# unique_pairs_non_explore_3_all = unique_pairs_non_explore_3

# # Plot the total number of unique state-action pairs across all configurations
# def plot_unique_state_action_pairs(unique_pairs, title):
#     plt.bar(range(len(unique_pairs)), unique_pairs.values(), color='skyblue', alpha=0.8)
#     plt.xticks(range(len(unique_pairs)), unique_pairs.keys())
#     plt.xlabel('Configuration')
#     plt.ylabel('Total Unique State-Action Pairs')
#     plt.title(title)
#     plt.show()

# plot_unique_state_action_pairs({
#     "Exploring Starts": unique_pairs_explore_all,
#     "Non-exploring Starts (𝜖 = 1/k)": unique_pairs_non_explore_1_all,
#     "Non-exploring Starts (𝜖 = e^(-k/1000))": unique_pairs_non_explore_2_all,
#     "Non-exploring Starts (𝜖 = e^(-k/10000))": unique_pairs_non_explore_3_all
# }, "Total Unique State-Action Pairs Across Configurations")

# def get_best_action(player_sum, dealer_card, ace, q_values):
#     state = (player_sum, dealer_card, ace)
#     if state in q_values:
#         hit_value = q_values[state].get('HIT', float('-inf'))
#         stand_value = q_values[state].get('STAND', float('-inf'))
#         if hit_value > stand_value:
#             return 'HIT'
#         elif stand_value > hit_value:
#             return 'STAND'
#         else:
#             return 'HIT' if random.random() < 0.5 else 'STAND'  # Randomly choose between HIT and STAND if values are equal
#     else:
#         return 'HIT'  # Default to HIT if state not found in Q-values


# def build_strategy_tables(q_values):
#     strategy_tables = []

#     for ace in [True, False]:
#         strategy_table = {}

#         for player_sum in range(20, 11, -1):
#             strategy_table[player_sum] = {}

#             for dealer_card in range(2, 11):
#                 best_action = get_best_action(player_sum, dealer_card, ace, q_values)
#                 strategy_table[player_sum][dealer_card] = best_action

#             # Handle Ace as 11
#             if ace:
#                 best_action = get_best_action(player_sum, 'A', ace, q_values)
#                 strategy_table[player_sum]['A'] = best_action

#         strategy_tables.append(strategy_table)

#     return strategy_tables

# # Example usage:
# strategy_tables_explore = build_strategy_tables(q_values_explore)
# strategy_tables_non_explore_1 = build_strategy_tables(q_values_non_explore_1)
# strategy_tables_non_explore_2 = build_strategy_tables(q_values_non_explore_2)
# strategy_tables_non_explore_3 = build_strategy_tables(q_values_non_explore_3)



# def print_strategy_table(strategy_table, has_ace):
#     ace_status = "with Ace" if has_ace else "without Ace"
#     print(f"Blackjack Strategy Table {ace_status}:")
#     print("Dealer's Card: ", end="")
#     for dealer_card in range(2, 11):
#         print(f"{dealer_card} ", end="")
#     print("A")
#     for player_sum in range(20, 11, -1):
#         print(f"{player_sum}: ", end="")
#         for dealer_card in range(2, 11):
#             print(strategy_table[player_sum].get(dealer_card, ''), end=" ")
#         print(strategy_table[player_sum].get('A', ''))
#     print()

# # Print strategy tables for each configuration
# for i, strategy_tables in enumerate([strategy_tables_explore, strategy_tables_non_explore_1, strategy_tables_non_explore_2, strategy_tables_non_explore_3]):
#     print(f"Configuration {i+1}:")
#     for ace in [True, False]:
#         print_strategy_table(strategy_tables[ace], ace)









