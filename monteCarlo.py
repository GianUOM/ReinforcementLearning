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
        total = 0
        has_ace = False
        for card in cards:
            if card.rank in ['J', 'Q', 'K']:
                total += 10
            elif card.rank == 'A':
                total += 11
                has_ace = True
            else:
                total += int(card.rank)
        if total > 21 and has_ace:
            total -= 10
        return total

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

def generate_state(player_sum, dealer_card, has_ace):
    return (player_sum, dealer_card, has_ace)

def choose_action(q_values, state, epsilon):
    if state not in q_values or not q_values[state]:
        return np.random.choice(['HIT', 'STAND'])
    if random.random() < epsilon:
        return np.random.choice(['HIT', 'STAND'])
    else:
        return max(q_values[state], key=q_values[state].get)

def monte_carlo_exploring_starts(num_episodes):
    q_values = {}
    for episode in range(1, num_episodes + 1):
        round = BlackjackRound()
        round.start()
        state = generate_state(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
        episode_actions = []
        episode_states = []
        episode_rewards = []

        if round.get_sum(round.player_cards) in range(12, 21):
            action = np.random.choice(['HIT', 'STAND'])
        else:
            action = choose_action(q_values, state, 1)  

        episode_actions.append(action)
        episode_states.append(state)

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
                elif player_sum > dealer_sum:
                    episode_rewards.append(1)
                elif player_sum < dealer_sum:
                    episode_rewards.append(-1)
                else:
                    episode_rewards.append(0)
                break

            state = generate_state(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
            episode_states.append(state)

            epsilon = 1 / episode
            if round.get_sum(round.player_cards) in range(12, 21):
                action = choose_action(q_values, state, epsilon)
            else:
                action = choose_action(q_values, state, 1)  
            episode_actions.append(action)

        G = 0
        for t in reversed(range(len(episode_rewards))):
            G += episode_rewards[t]
            state = episode_states[t]
            action = episode_actions[t]
            if state not in q_values:
                q_values[state] = {}
            if action not in q_values[state]:
                q_values[state][action] = 0
            q_values[state][action] += (G - q_values[state][action]) / (t + 1)

    return q_values

def monte_carlo_non_exploring_starts(num_episodes, alpha, epsilon_func):
    q_values = {}
    for episode in range(num_episodes):
        round = BlackjackRound()
        round.start()
        state = generate_state(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
        episode_actions = []
        episode_rewards = []

        action = choose_action(q_values, state, epsilon_func(episode + 1))  
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

            next_state = generate_state(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
            state = next_state

            action = choose_action(q_values, state, epsilon_func(episode + 1))  
            episode_actions.append(action)

        for i in range(len(episode_actions)):
            state_action = (state, episode_actions[i])
            G = sum(episode_rewards[i:])
            if state not in q_values:
                q_values[state] = {}
            if episode_actions[i] not in q_values[state]:
                q_values[state][episode_actions[i]] = 0
            q_values[state][episode_actions[i]] += alpha * (G - q_values[state][episode_actions[i]])

    return q_values

def run_episodes_and_extract_info(agent_function, num_episodes, alpha, epsilon_func):
    q_values = {}
    episode_results = {'Win': [], 'Loss': [], 'Draw': []}
    unique_state_action_pairs = set()
    action_counts = {}
    wins_per_episode = []
    losses_per_episode = []
    draws_per_episode = []

    for episode in range(1, num_episodes + 1):
        round = BlackjackRound()
        round.start()
        state = generate_state(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
        episode_actions = []
        episode_rewards = []

        action = choose_action(q_values, state, epsilon_func(episode))  
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

            next_state = generate_state(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
            state = next_state

            action = choose_action(q_values, state, epsilon_func(episode))  
            episode_actions.append(action)

        for i in range(len(episode_actions)):
            state_action = (state, episode_actions[i])
            G = sum(episode_rewards[i:])
            if state not in q_values:
                q_values[state] = {}
            if episode_actions[i] not in q_values[state]:
                q_values[state][episode_actions[i]] = 0
            q_values[state][episode_actions[i]] += alpha * (G - q_values[state][episode_actions[i]])

        episode_result = round.get_outcome()
        episode_results[episode_result].append(1)

        if episode_result == 'Win':
            wins_per_episode.append(1)
            losses_per_episode.append(0)
            draws_per_episode.append(0)
        elif episode_result == 'Loss':
            wins_per_episode.append(0)
            losses_per_episode.append(1)
            draws_per_episode.append(0)
        else:
            wins_per_episode.append(0)
            losses_per_episode.append(0)
            draws_per_episode.append(1)

        unique_state_action_pairs.add(state_action)

        if (state, action) in action_counts:
            action_counts[(state, action)] += 1
        else:
            action_counts[(state, action)] = 1

        if episode % 1000 == 0:
            print(f"Episodes {episode-999}-{episode}: Wins - {sum(episode_results['Win'])}, Losses - {sum(episode_results['Loss'])}, Draws - {sum(episode_results['Draw'])}")
            episode_results = {'Win': [], 'Loss': [], 'Draw': []}

    return wins_per_episode, losses_per_episode, draws_per_episode, sum(episode_results['Win']), sum(episode_results['Loss']), sum(episode_results['Draw']), len(unique_state_action_pairs), action_counts, q_values



num_episodes = 100000
alpha = 0.05
wins_per_episode_explore, losses_per_episode_explore, draws_per_episode_explore, wins_explore, losses_explore, draws_explore, unique_pairs_explore, action_counts_explore, q_values_explore = run_episodes_and_extract_info(monte_carlo_exploring_starts, num_episodes, alpha, lambda k: 1 / k)

num_episodes = 100000
alpha = 0.05
wins_per_episode1, losses_per_episode1, draws_per_episode1, wins_non_explore_1, losses_non_explore_1, draws_non_explore_1, unique_pairs_non_explore_1, action_counts_non_explore_1, q_values_non_explore_1 = run_episodes_and_extract_info(monte_carlo_non_exploring_starts, num_episodes, alpha, lambda k: 1 / k)

num_episodes = 100000
alpha = 0.05
wins_per_episode2, losses_per_episode2, draws_per_episode2, wins_non_explore_2, losses_non_explore_2, draws_non_explore_2, unique_pairs_non_explore_2, action_counts_non_explore_2, q_values_non_explore_2 = run_episodes_and_extract_info(monte_carlo_non_exploring_starts, num_episodes, alpha, lambda k: np.exp(-k / 1000))

num_episodes = 100000
alpha = 0.05
wins_per_episode3, losses_per_episode3, draws_per_episode3, wins_non_explore_3, losses_non_explore_3, draws_non_explore_3, unique_pairs_non_explore_3, action_counts_non_explore_3, q_values_non_explore_3 = run_episodes_and_extract_info(monte_carlo_non_exploring_starts, num_episodes, alpha, lambda k: np.exp(-k / 10000))

print('-----------------------------------------------------------------------')
print("\nResults for Exploring Starts:\n")
print("Wins: ", wins_explore)
print("Losses: ", losses_explore)
print("Draws: ", draws_explore)
print("Unique state-action pairs explored: ", unique_pairs_explore)
print("\nCounts of state-action pair selections:")
for state_action, count in action_counts_explore.items():
    print(f"State-Action Pair: {state_action}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in q_values_explore.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")

print('-----------------------------------------------------------------------')
print("\nResults for Non-exploring Starts with 𝜖 = 1/k:\n")
print("Wins: ", wins_non_explore_1)
print("Losses: ", losses_non_explore_1)
print("Draws: ", draws_non_explore_1)
print("Unique state-action pairs explored: ", unique_pairs_non_explore_1)
print("\nCounts of state-action pair selections:")
for state_action, count in action_counts_non_explore_1.items():
    print(f"State-Action Pair: {state_action}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in q_values_non_explore_1.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")

print('-----------------------------------------------------------------------')
print("\nResults for Non-exploring Starts with 𝜖 = e^(-k/1000):\n")
print("Wins: ", wins_non_explore_2)
print("Losses: ", losses_non_explore_2)
print("Draws: ", draws_non_explore_2)
print("Unique state-action pairs explored: ", unique_pairs_non_explore_2)
print("\nCounts of state-action pair selections:")
for state_action, count in action_counts_non_explore_2.items():
    print(f"State-Action Pair: {state_action}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in q_values_non_explore_2.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")

print('-----------------------------------------------------------------------')
print("\nResults for Non-exploring Starts with 𝜖 = e^(-k/10000):\n")
print("Wins: ", wins_non_explore_3)
print("Losses: ", losses_non_explore_3)
print("Draws: ", draws_non_explore_3)
print("Unique state-action pairs explored: ", unique_pairs_non_explore_3)
print("\nCounts of state-action pair selections:")
for state_action, count in action_counts_non_explore_3.items():
    print(f"State-Action Pair: {state_action}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in q_values_non_explore_3.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")

def plot_results(wins_per_episode, losses_per_episode, draws_per_episode, num_episodes, title):
    interval = 1000
    wins_interval = []
    losses_interval = []
    draws_interval = []

    for i in range(0, num_episodes, interval):
        wins_interval.append(sum(wins_per_episode[i:i+interval]))
        losses_interval.append(sum(losses_per_episode[i:i+interval]))
        draws_interval.append(sum(draws_per_episode[i:i+interval]))

    plt.figure(figsize=(10, 6))
    plt.plot(range(0, num_episodes, interval), wins_interval, label='Wins', marker='o', linestyle='-', color='blue')
    plt.plot(range(0, num_episodes, interval), losses_interval, label='Losses', marker='o', linestyle='-', color='red')
    plt.plot(range(0, num_episodes, interval), draws_interval, label='Draws', marker='o', linestyle='-', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Counts')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_action_counts(action_counts, title):
    valid_player_sums = list(range(12, 21))  
    valid_dealer_cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']

    filtered_action_counts = {
        state_action: count for state_action, count in action_counts.items()
        if state_action[0][0] in valid_player_sums and state_action[0][1] in valid_dealer_cards
    }

    sorted_action_counts = sorted(filtered_action_counts.items(), key=lambda x: x[1], reverse=True)
    states_actions, counts = zip(*sorted_action_counts)

    plt.figure(figsize=(20, 10))  
    plt.bar(range(len(states_actions)), counts)
    plt.xlabel('State-Action Pairs')
    plt.ylabel('Counts') 
    plt.title(title)
    plt.tight_layout()  
    plt.show()

def plot_unique_state_action_pairs(configurations, unique_pairs_counts):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(configurations, unique_pairs_counts, color=['blue', 'orange', 'green', 'red'])

    
    max_unique_pairs_index = unique_pairs_counts.index(max(unique_pairs_counts))
    bars[max_unique_pairs_index].set_color('purple')

    plt.xlabel('Algorithm Configurations')
    plt.ylabel('Number of Unique State-Action Pairs')
    plt.yticks(np.arange(0, max(unique_pairs_counts) + 50, 50))  
    plt.title('Total Number of Unique State-Action Pairs Across Configurations')
    plt.grid(True)
    plt.show()

def build_strategy_table(q_values, has_ace):
    dealer_cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']
    player_sums = list(range(20, 11, -1))
    strategy_table = {sum_: {dealer_card: '' for dealer_card in dealer_cards} for sum_ in player_sums}

    for player_sum in player_sums:
        for dealer_card in dealer_cards:
            state = (player_sum, dealer_card, has_ace)
            if state in q_values:
                best_action = max(q_values[state], key=q_values[state].get)
                strategy_table[player_sum][dealer_card] = 'H' if best_action == 'HIT' else 'S'
            else:
                strategy_table[player_sum][dealer_card] = 'N/A'  
    
    return strategy_table

def print_strategy_table(strategy_table, title):
    print(title)
    print("Player Sum | " + " | ".join(strategy_table[20].keys()))
    print("-" * (11 * len(strategy_table[20].keys()) + 12))
    for player_sum, actions in strategy_table.items():
        actions_str = " | ".join(actions[dealer_card] for dealer_card in actions)
        print(f"    {player_sum}    | {actions_str}")

def analyze_last_10000_episodes(wins_per_episode, losses_per_episode, draws_per_episode, title):
    total_wins_last_10000 = sum(wins_per_episode[-10000:])
    total_losses_last_10000 = sum(losses_per_episode[-10000:])
    total_draws_last_10000 = sum(draws_per_episode[-10000:])
    print(f"{title} - Last 10000 Episodes: Wins = {total_wins_last_10000}, Losses = {total_losses_last_10000}, Draws = {total_draws_last_10000}")
    return total_wins_last_10000, total_losses_last_10000, total_draws_last_10000

def calculate_dealer_advantage(total_wins, total_losses):
    mean_wins = total_wins / 10000
    mean_losses = total_losses / 10000
    return (mean_losses - mean_wins) / (mean_losses + mean_wins)

def compare_dealer_advantage(configurations, advantages):
    min_advantage_index = advantages.index(min(advantages))
    min_advantage_config = configurations[min_advantage_index]

    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(configurations, advantages, color=['blue', 'orange', 'green', 'red'])

    plt.xlabel('Algorithm Configurations')
    plt.ylabel("Dealer's Advantage")
    plt.title('Dealer Advantage Across Different Algorithm Configurations')
    plt.grid(True)
    plt.show()

    print(f"\nThe algorithm that minimizes the dealer advantage the most is: {min_advantage_config}")

plot_results(wins_per_episode_explore, losses_per_episode_explore, draws_per_episode_explore, num_episodes, 'Exploring Starts')
plot_results(wins_per_episode1, losses_per_episode1, draws_per_episode1, num_episodes, "Non-exploring Starts (1/k)")
plot_results(wins_per_episode2, losses_per_episode2, draws_per_episode2, num_episodes, "Non-exploring Starts (e^(-k/1000))")
plot_results(wins_per_episode1, losses_per_episode3, draws_per_episode3, num_episodes, "Non-exploring Starts (e^(-k/10000))")
plot_action_counts(action_counts_explore, 'Counts of State-Action Pairs: Exploring Starts')
plot_action_counts(action_counts_non_explore_1, 'Counts of State-Action Pairs: Non-exploring Starts (1/k)')
plot_action_counts(action_counts_non_explore_2, 'Counts of State-Action Pairs: Non-exploring Starts (e^(-k/1000))')
plot_action_counts(action_counts_non_explore_3, 'Counts of State-Action Pairs: Non-exploring Starts (e^(-k/10000))')
configurations = ['Exploring Starts', 'Non-exploring Starts (1/k)', 'Non-exploring Starts (e^(-k/1000))', 'Non-exploring Starts (e^(-k/10000))']
unique_pairs_counts = [unique_pairs_explore, unique_pairs_non_explore_1, unique_pairs_non_explore_2, unique_pairs_non_explore_3]
plot_unique_state_action_pairs(configurations, unique_pairs_counts)

configs = [
    ("Exploring Starts", q_values_explore),
    ("Non-exploring Starts with 𝜖 = 1/k", q_values_non_explore_1),
    ("Non-exploring Starts with 𝜖 = e^(-k/1000)", q_values_non_explore_2),
    ("Non-exploring Starts with 𝜖 = e^(-k/10000)", q_values_non_explore_3)
]

for config_name, q_values in configs:
    strategy_table_ace = build_strategy_table(q_values, has_ace=True)
    print_strategy_table(strategy_table_ace, f"Strategy Table with Ace as 11 - {config_name}")
    
    strategy_table_no_ace = build_strategy_table(q_values, has_ace=False)
    print_strategy_table(strategy_table_no_ace, f"Strategy Table without Ace as 11 - {config_name}")


total_wins_explore_last_10000, total_losses_explore_last_10000, total_draws_explore_last_10000 = analyze_last_10000_episodes(wins_per_episode_explore, losses_per_episode_explore, draws_per_episode_explore, 'Exploring Starts')
total_wins_non_explore_1_last_10000, total_losses_non_explore_1_last_10000, total_draws_non_explore_1_last_10000 = analyze_last_10000_episodes(wins_per_episode1, losses_per_episode1, draws_per_episode1, 'Non-exploring Starts with 𝜖 = 1/k')
total_wins_non_explore_2_last_10000, total_losses_non_explore_2_last_10000, total_draws_non_explore_2_last_10000 = analyze_last_10000_episodes(wins_per_episode2, losses_per_episode2, draws_per_episode2, 'Non-exploring Starts with 𝜖 = e^(-k/1000)')
total_wins_non_explore_3_last_10000, total_losses_non_explore_3_last_10000, total_draws_non_explore_3_last_10000 = analyze_last_10000_episodes(wins_per_episode3, losses_per_episode3, draws_per_episode3, 'Non-exploring Starts with 𝜖 = e^(-k/10000)')

dealer_advantage_explore = calculate_dealer_advantage(total_wins_explore_last_10000, total_losses_explore_last_10000)
dealer_advantage_non_explore_1 = calculate_dealer_advantage(total_wins_non_explore_1_last_10000, total_losses_non_explore_1_last_10000)
dealer_advantage_non_explore_2 = calculate_dealer_advantage(total_wins_non_explore_2_last_10000, total_losses_non_explore_2_last_10000)
dealer_advantage_non_explore_3 = calculate_dealer_advantage(total_wins_non_explore_3_last_10000, total_losses_non_explore_3_last_10000)

print(f"Exploring Starts - Dealer's Advantage: {dealer_advantage_explore:.4f}")
print(f"Non-exploring Starts with 𝜖 = 1/k - Dealer's Advantage: {dealer_advantage_non_explore_1:.4f}")
print(f"Non-exploring Starts with 𝜖 = e^(-k/1000) - Dealer's Advantage: {dealer_advantage_non_explore_2:.4f}")
print(f"Non-exploring Starts with 𝜖 = e^(-k/10000) - Dealer's Advantage: {dealer_advantage_non_explore_3:.4f}")

configurations = ['Exploring Starts', 'Non-exploring Starts (1/k)', 'Non-exploring Starts (e^(-k/1000))', 'Non-exploring Starts (e^(-k/10000))']
advantages = [dealer_advantage_explore, dealer_advantage_non_explore_1, dealer_advantage_non_explore_2, dealer_advantage_non_explore_3]

compare_dealer_advantage(configurations, advantages)
