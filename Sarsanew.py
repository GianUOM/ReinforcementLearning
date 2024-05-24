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
        

class RLAgentSARSA:
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

    def update_q_values(self, state, action, reward, next_state, next_action, alpha):
        if state not in self.q_values:
            self.q_values[state] = {}
        if next_state not in self.q_values:
            self.q_values[next_state] = {}

        old_q_value = self.q_values[state].get(action, 0)
        next_q_value = self.q_values[next_state].get(next_action, 0)

        td_target = reward + next_q_value
        td_error = td_target - old_q_value

        new_q_value = old_q_value + alpha * td_error
        self.q_values[state][action] = new_q_value

    def update_action_counts(self, state, action):
        if state not in self.action_counts:
            self.action_counts[state] = {}
        if action not in self.action_counts[state]:
            self.action_counts[state][action] = 0
        self.action_counts[state][action] += 1
epsilon_func = 0.1
def sarsa(num_of_episodes, alpha, epsilon):
    agent = RLAgentSARSA()
    for episode in range(num_of_episodes):
        round = BlackjackRound()
        round.start()
        state = agent.generate_agent_state(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
        episode_actions = []
        episode_rewards = []

        # Initial action selection
        action = agent.choose_action(state, epsilon)

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

            action = agent.choose_action(state, epsilon_func(episode+1))  
            episode_actions.append(action)

            agent.update_q_values(state, action, episode_rewards, next_state, alpha)

            state = next_state      
        
    return agent.q_values




def run_episodes(agent_function, num_of_episodes, alpha, epsilon_func):
    agent = RLAgentSARSA()
    episode_results = {'Win': [], 'Loss': [], 'Draw': []}
    unique_state_action_pairs = set()
    action_counts = {}
    q_values = {}
    wins_per_episode = []

    for episode in range(1, num_of_episodes + 1):
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
        else:
            wins_per_episode.append(0)

        unique_state_action_pairs.add((state, action))

        if (state, action) in action_counts:
            action_counts[(state, action)] += 1
        else:
            action_counts[(state, action)] = 1

        if episode % 1000 == 0:
            print(f"Episodes {episode-999}-{episode}: Wins - {sum(episode_results['Win'])}, Losses - {sum(episode_results['Loss'])}, Draws - {sum(episode_results['Draw'])}")
            episode_results = {'Win': [], 'Loss': [], 'Draw': []}

    return wins_per_episode, sum(episode_results['Win']), sum(episode_results['Loss']), sum(episode_results['Draw']), len(unique_state_action_pairs), action_counts, q_values


def count_unique_state_action_pairs(action_counts):
    valid_player_sums = list(range(12, 21))  # Valid player sums from 12 to 20
    valid_dealer_cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']
    
    unique_pairs = set()
    
    for (state, action) in action_counts.keys():
        player_sum, dealer_card, has_ace = state
        if player_sum in valid_player_sums and dealer_card in valid_dealer_cards:
            unique_pairs.add((state, action))
    
    return len(unique_pairs)



# Configuration 1: epsilon = 0.1
num_of_episodes = 100000
alpha = 0.1
wins_per_episode_config1, wins_config1, losses_config1, draws_config1, unique_pairs_config1, action_counts_config1, q_values_config1 = run_episodes(sarsa, num_of_episodes, alpha, lambda k: 0.1)
unique_pairs_config1_refined = count_unique_state_action_pairs(action_counts_config1)

# Configuration 2: epsilon = 1/k
num_of_episodes = 100000
alpha = 0.1
wins_per_episode_config2, wins_config2, losses_config2, draws_config2, unique_pairs_config2, action_counts_config2, q_values_config2 = run_episodes(sarsa, num_of_episodes, alpha, lambda k: 1 / k)
unique_pairs_config2_refined = count_unique_state_action_pairs(action_counts_config2)

# Configuration 3: epsilon = e^(-k/1000)
num_of_episodes = 100000
alpha = 0.1
wins_per_episode_config3, wins_config3, losses_config3, draws_config3, unique_pairs_config3, action_counts_config3, q_values_config3 = run_episodes(sarsa, num_of_episodes, alpha, lambda k: np.exp(-k / 1000))
unique_pairs_config3_refined = count_unique_state_action_pairs(action_counts_config3)

# Configuration 4: epsilon = e^(-k/10000)
num_of_episodes = 100000
alpha = 0.1
wins_per_episode_config4, wins_config4, losses_config4, draws_config4, unique_pairs_config4, action_counts_config4, q_values_config4 = run_episodes(sarsa, num_of_episodes, alpha, lambda k: np.exp(-k / 10000))
unique_pairs_config4_refined = count_unique_state_action_pairs(action_counts_config4)

# Print results for config1
print('-----------------------------------------------------------------------')
print("\nResults for epsilon = 0.1:\n")
print("Wins: ", wins_config1)
print("Losses: ", losses_config1)
print("Draws: ", draws_config1)
print("Unique state-action pairs explored: ", unique_pairs_config1_refined)
print("\nCounts of state-action pair selections:")
for state_action, count in action_counts_config1.items():
    print(f"State-Action Pair: {state_action}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in q_values_config1.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")

# Print results for config2
print('-----------------------------------------------------------------------')
print("\nResults for epsilon = 1/k:\n")
print("Wins: ", wins_config2)
print("Losses: ", losses_config2)
print("Draws: ", draws_config2)
print("Unique state-action pairs explored: ", unique_pairs_config2_refined)
print("\nCounts of state-action pair selections:")
for state_action, count in action_counts_config2.items():
    print(f"State-Action Pair: {state_action}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in q_values_config2.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")

# Print results for config3
print('-----------------------------------------------------------------------')
print("\nResults for epsilon = e^(-k/1000):\n")
print("Wins: ", wins_config3)
print("Losses: ", losses_config3)
print("Draws: ", draws_config3)
print("Unique state-action pairs explored: ", unique_pairs_config3_refined)
print("\nCounts of state-action pair selections:")
for state_action, count in action_counts_config3.items():
    print(f"State-Action Pair: {state_action}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in q_values_config3.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")

# Print results for config4
print('-----------------------------------------------------------------------')
print("\nResults for epsilon = e^(-k/10000):\n")
print("Wins: ", wins_config4)
print("Losses: ", losses_config4)
print("Draws: ", draws_config4)
print("Unique state-action pairs explored: ", unique_pairs_config4_refined)
print("\nCounts of state-action pair selections:")
for state_action, count in action_counts_config4.items():
    print(f"State-Action Pair: {state_action}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in q_values_config4.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")


# Print results for config1
print('-----------------------------------------------------------------------')
print("\nResults for epsilon = 0.1:\n")
print("Wins: ", wins_config1)
print("Losses: ", losses_config1)
print("Draws: ", draws_config1)
print("Unique state-action pairs explored: ", unique_pairs_config1_refined)
print("\nCounts of state-action pair selections:")
for state_action, count in action_counts_config1.items():
    print(f"State-Action Pair: {state_action}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in q_values_config1.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")

# Print results for config2
print('-----------------------------------------------------------------------')
print("\nResults for epsilon = 1/k:\n")
print("Wins: ", wins_config2)
print("Losses: ", losses_config2)
print("Draws: ", draws_config2)
print("Unique state-action pairs explored: ", unique_pairs_config2_refined)
print("\nCounts of state-action pair selections:")
for state_action, count in action_counts_config2.items():
    print(f"State-Action Pair: {state_action}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in q_values_config2.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")

# Print results for config3
print('-----------------------------------------------------------------------')
print("\nResults for epsilon = e^(-k/1000):\n")
print("Wins: ", wins_config3)
print("Losses: ", losses_config3)
print("Draws: ", draws_config3)
print("Unique state-action pairs explored: ", unique_pairs_config3_refined)
print("\nCounts of state-action pair selections:")
for state_action, count in action_counts_config3.items():
    print(f"State-Action Pair: {state_action}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in q_values_config3.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")

# Print results for config4
print('-----------------------------------------------------------------------')
print("\nResults for epsilon = e^(-k/10000):\n")
print("Wins: ", wins_config4)
print("Losses: ", losses_config4)
print("Draws: ", draws_config4)
print("Unique state-action pairs explored: ", unique_pairs_config4_refined)
print("\nCounts of state-action pair selections:")
for state_action, count in action_counts_config4.items():
    print(f"State-Action Pair: {state_action}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in q_values_config4.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")


interval = 20000
wins_interval = [0]
losses_interval = [0]
draws_interval = [0]

for i in range(interval, num_of_episodes + interval, interval):
    wins_interval.append(sum(wins_per_episode_config1[:i]))
    losses_interval.append(i - sum(wins_per_episode_config1[:i]))
    draws_interval.append(0)

plt.figure(figsize=(10, 6))
plt.plot(range(0, num_of_episodes + 1, interval), wins_interval, label='Wins', marker='o', color='blue')
plt.plot(range(0, num_of_episodes + 1, interval), losses_interval, label='Losses', marker='o', color='red')
plt.plot(range(0, num_of_episodes + 1, interval), draws_interval, label='Draws', marker='o', color='green')
plt.xlabel('Number of Episodes')
plt.ylabel('Results')
plt.title('Results for epsilon = 0.1 Configuration')
plt.legend()
plt.grid(True)
plt.show()


interval = 20000
wins_interval = [0]
losses_interval = [0]
draws_interval = [0]

for i in range(interval, num_of_episodes + interval, interval):
    wins_interval.append(sum(wins_per_episode_config2[:i]))
    losses_interval.append(i - sum(wins_per_episode_config2[:i]))
    draws_interval.append(0)

plt.figure(figsize=(10, 6))
plt.plot(range(0, num_of_episodes + 1, interval), wins_interval, label='Wins', marker='o', color='blue')
plt.plot(range(0, num_of_episodes + 1, interval), losses_interval, label='Losses', marker='o', color='red')
plt.plot(range(0, num_of_episodes + 1, interval), draws_interval, label='Draws', marker='o', color='green')
plt.xlabel('Number of Episodes')
plt.ylabel('Results')
plt.title('Results for epsilon = 1/k Configuration')
plt.legend()
plt.grid(True)
plt.show()


interval = 20000
wins_interval = [0]
losses_interval = [0]
draws_interval = [0]

for i in range(interval, num_of_episodes + interval, interval):
    wins_interval.append(sum(wins_per_episode_config3[:i]))
    losses_interval.append(i - sum(wins_per_episode_config3[:i]))
    draws_interval.append(0)

plt.figure(figsize=(10, 6))
plt.plot(range(0, num_of_episodes + 1, interval), wins_interval, label='Wins', marker='o', color='blue')
plt.plot(range(0, num_of_episodes + 1, interval), losses_interval, label='Losses', marker='o', color='red')
plt.plot(range(0, num_of_episodes + 1, interval), draws_interval, label='Draws', marker='o', color='green')
plt.xlabel('Number of Episodes')
plt.ylabel('Results')
plt.title('Results for epsilon = e^(-k/1000) Configuration')
plt.legend()
plt.grid(True)
plt.show()


interval = 20000
wins_interval = [0]
losses_interval = [0]
draws_interval = [0]

for i in range(interval, num_of_episodes + interval, interval):
    wins_interval.append(sum(wins_per_episode_config4[:i]))
    losses_interval.append(i - sum(wins_per_episode_config4[:i]))
    draws_interval.append(0)

plt.figure(figsize=(10, 6))
plt.plot(range(0, num_of_episodes + 1, interval), wins_interval, label='Wins', marker='o', color='blue')
plt.plot(range(0, num_of_episodes + 1, interval), losses_interval, label='Losses', marker='o', color='red')
plt.plot(range(0, num_of_episodes + 1, interval), draws_interval, label='Draws', marker='o', color='green')
plt.xlabel('Number of Episodes')
plt.ylabel('Results')
plt.title('Results for epsilon = e^(-k/10000) Configuration')
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

# Plotting each configuration
plot_action_counts(action_counts_config1, 'Counts of State-Action Pairs: epsilon = 0.1')
plot_action_counts(action_counts_config2, 'Counts of State-Action Pairs: (epsilon = 1/k)')
plot_action_counts(action_counts_config3, 'Counts of State-Action Pairs: (epsilon = e^(-k/1000))')
plot_action_counts(action_counts_config4, 'Counts of State-Action Pairs: (epsilon = e^(-k/10000))')

configurations = ['epsilon = 0.1', 'epsilon = 1/k', 'epsilon = (e^(-k/1000))', 'epsilon = (e^(-k/10000))']
unique_pairs_counts = [unique_pairs_config1_refined, unique_pairs_config2_refined, unique_pairs_config3_refined, unique_pairs_config4_refined]

plt.figure(figsize=(10, 6))
plt.bar(configurations, unique_pairs_counts, color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Algorithm Configurations')
plt.ylabel('Number of Unique State-Action Pairs')
plt.yticks(np.arange(0, max(unique_pairs_counts) + 50, 50))  # Set y-axis interval to 50
plt.title('Total Number of Unique State-Action Pairs Across Configurations')
plt.show()

def build_strategy_table(q_values, has_ace):
    dealer_cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']
    player_sums = list(range(20, 11, -1))
    
    # Initialize the strategy table
    strategy_table = {sum_: {dealer_card: '' for dealer_card in dealer_cards} for sum_ in player_sums}

    # Fill the strategy table
    for player_sum in player_sums:
        for dealer_card in dealer_cards:
            state = (player_sum, dealer_card, has_ace)
            if state in q_values:
                best_action = max(q_values[state], key=q_values[state].get)
                strategy_table[player_sum][dealer_card] = 'H' if best_action == 'HIT' else 'S'
            else:
                strategy_table[player_sum][dealer_card] = 'N/A'  # If state is not in q_values
    
    return strategy_table

def print_strategy_table(strategy_table, title):
    print(title)
    print("Player Sum | " + " | ".join(strategy_table[20].keys()))
    print("-" * (11 * len(strategy_table[20].keys()) + 12))
    for player_sum, actions in strategy_table.items():
        actions_str = " | ".join(actions[dealer_card] for dealer_card in actions)
        print(f"    {player_sum}    | {actions_str}")

# Generate strategy tables for each configuration
configs = [
    ("epsilon = 0.1", q_values_config1),
    ("epsilon = 1/k", q_values_config2),
    ("epsilon = e^(-k/1000)", q_values_config3),
    ("epsilon = e^(-k/10000)", q_values_config4)
]

for config_name, q_values in configs:
    # Strategy tables for player with an Ace as 11
    strategy_table_ace = build_strategy_table(q_values, has_ace=True)
    print_strategy_table(strategy_table_ace, f"Strategy Table with Ace as 11 - {config_name}")
    
    # Strategy tables for player without an Ace as 11
    strategy_table_no_ace = build_strategy_table(q_values, has_ace=False)
    print_strategy_table(strategy_table_no_ace, f"Strategy Table without Ace as 11 - {config_name}")

