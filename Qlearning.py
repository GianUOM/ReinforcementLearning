import random
import numpy as np
import math
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
        if self.cards:
            return self.cards.pop()
        else:
            print("Deck is empty!")
            return None
    
class Action:
    HIT = 'Hit'
    STAND = 'Stand'

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
        while total > 21 and has_ace:
            total -= 10
            has_ace = False
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

    def get_state(self):
        player_sum = self.get_sum(self.player_cards)
        dealer_card = self.dealer_cards[0].rank
        return (player_sum, dealer_card)
    
    def get_reward(self):
        outcome = self.get_outcome()
        if outcome == 'Win':
            return 1
        elif outcome == 'Loss':
            return -1
        else:
            return 0


class QLearning:
    def __init__(self, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma  
        self.Q = {}  # Q-table

    def choose_action(self, state):
        if state not in self.q_values or not self.q_values[state]:
            return np.random.choice(['HIT', 'STAND'])

        if np.random.rand() < self.epsilon:
            return np.random.choice(['HIT', 'STAND'])
        else:
            return max(self.q_values[state], key=self.q_values[state].get)

    def initialize_Q_values(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['Spades', 'Clubs', 'Hearts', 'Diamonds']
        for player_sum in range(12, 21):
            for dealer_card in ranks[1:]:
                state = (player_sum, dealer_card)
                self.Q[state] = {Action.HIT: 0, Action.STAND: 0}

    def get_action(self, state):
        if state not in self.Q:
            self.Q[state] = {Action.HIT: 0, Action.STAND: 0}
        
        if random.random() < self.epsilon:
            return random.choice([Action.HIT, Action.STAND])
        else:
            max_q_value_hit = self.Q[state][Action.HIT]
            max_q_value_stand = self.Q[state][Action.STAND]
            if max_q_value_hit > max_q_value_stand:
                return Action.HIT
            elif max_q_value_hit < max_q_value_stand:
                return Action.STAND
            else:
                return random.choice([Action.HIT, Action.STAND])

    def update_Q(self, state, action, reward, next_state, next_action):
        if state not in self.Q:
            self.Q[state] = {Action.HIT: 0, Action.STAND: 0}
        if next_state not in self.Q:
            self.Q[next_state] = {Action.HIT: 0, Action.STAND: 0}
        td_target = reward + self.gamma * max(self.Q[next_state].values())
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def update_q_values(self, state, action, reward, next_state):
        if state not in self.q_values:
            self.q_values[state] = {'HIT': 0, 'STAND': 0}

        if next_state not in self.q_values:
            self.q_values[next_state] = {'HIT': 0, 'STAND': 0}

        td_target = reward + self.gamma * max(self.Q[next_state].values())
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

        max_next_action_value = max(self.q_values[next_state].values())

        self.q_values[state][action] += self.alpha * (reward + self.gamma * max_next_action_value - self.q_values[state][action])



alpha = 0.1
gamma = 1.0
episode = 1
# Different values of epsilon
epsilon_values = [0.1, 1/episode, math.exp(-episode/1000), math.exp(-episode/10000)]

# Define lists to store data for each configuration
all_wins_data = []
all_losses_data = []
all_draws_data = []
all_total_unique_pairs = []
all_state_action_counts = []

configuration = 1
for epsilon in epsilon_values:
    alpha = 0.1
    gamma = 1.0
    print(f"Running configuration {configuration}")
    configuration+=1
    q_learning_agent = QLearning(epsilon=epsilon, alpha=alpha, gamma=gamma)

    episode_rewards = []
    win_count = 0
    loss_count = 0
    draw_count = 0
    state_action_counts = {}
    unique_state_action_pairs = set()


    wins_data = []
    losses_data = []
    draws_data = []
    state_action_counts = {}
    unique_state_action_pairs = set()


    for episode in range(100000):
        round = BlackjackRound()
        round.start()
        state = round.get_state()
        action = q_learning_agent.get_action(state)
        reward = 0

        while True:
            if action == Action.HIT:
                round.hit()
                if round.get_sum(round.player_cards) > 21:
                    reward = -1
                    next_state = round.get_state()
                    next_action = None
                    break
            elif action == Action.STAND:
                round.stand()
                reward = round.get_reward()
                next_state = round.get_state()
                next_action = None
                break

            next_state = round.get_state()
            next_action = q_learning_agent.get_action(next_state)
            
            q_learning_agent.update_Q(state, action, reward, next_state, next_action)
            
            state_action_pair = (state, action)
            if state_action_pair not in state_action_counts:
                state_action_counts[state_action_pair] = 1
                unique_state_action_pairs.add(state_action_pair)
            else:
                state_action_counts[state_action_pair] += 1

            state = next_state
            action = next_action

        if next_action is None:
            next_action = Action.STAND
        q_learning_agent.update_Q(state, action, reward, next_state, next_action)

        state_action_pair = (state, action)
        if state_action_pair not in state_action_counts:
            state_action_counts[state_action_pair] = 1
            unique_state_action_pairs.add(state_action_pair)
        else:
            state_action_counts[state_action_pair] += 1

        outcome = round.get_outcome()
        if outcome == 'Win':
            win_count += 1
        elif outcome == 'Loss':
            loss_count += 1
        elif outcome == 'Draw':
            draw_count += 1            

        episode_rewards.append(reward)
        
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            print(f"Episodes: {episode - 999}-{episode}, Average Reward: {avg_reward}, Wins: {win_count}, Losses: {loss_count}, Draws: {draw_count}")

            wins_data.append(win_count)
            losses_data.append(loss_count)
            draws_data.append(draw_count)
            win_count = 0
            loss_count = 0
            draw_count = 0

    # Output final statistics
    final_avg_reward = np.mean(episode_rewards)
    print(f"Final Average Reward for Epsilon {epsilon}: {final_avg_reward}")
    print(f"Number of unique state-action pairs explored for Epsilon {epsilon}: {len(unique_state_action_pairs)}")
    print("State-Action Pair Counts:")
    for pair, count in state_action_counts.items():
        print(f"State: {pair[0]}, Action: {pair[1]}, Count: {count}")
    print("Estimated Q values for Epsilon {epsilon}:")
    for pair in unique_state_action_pairs:
        print(f"State: {pair[0]}, Action: {pair[1]}, Q-value: {q_learning_agent.Q[pair[0]][pair[1]]}")

    all_wins_data.append(wins_data)
    all_losses_data.append(losses_data)
    all_draws_data.append(draws_data)
    all_total_unique_pairs.append(len(unique_state_action_pairs))
    all_state_action_counts.append(state_action_counts)

# Function to plot wins, losses, and draws
def plot_outcomes(wins_data, losses_data, draws_data, configuration):
    episodes = range(1000, 100001, 1000)
    plt.plot(episodes, wins_data, label='Wins')
    plt.plot(episodes, losses_data, label='Losses')
    plt.plot(episodes, draws_data, label='Draws')
    plt.xlabel('Episodes')
    plt.ylabel('Counts')
    plt.title(f'Configuration:{configuration}')
    plt.legend()
    plt.show()

# Plotting for each configuration
for configuration in range(len(epsilon_values)):
    plot_outcomes(all_wins_data[configuration], all_losses_data[configuration], all_draws_data[configuration], configuration + 1)

# Plot the counts of each unique state-action pair on a bar chart sorted by highest count first
for configuration in range(len(epsilon_values)):
    state_action_counts = all_state_action_counts[configuration]
    
    # Flatten state-action pairs and counts into lists
    counts = []
    for count in state_action_counts.values():
        counts.append(count)
    
    # Sort by counts
    sorted_counts = sorted(counts, reverse=True)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(sorted_counts)), sorted_counts)
    plt.xlabel('State-Action Pairs')
    plt.ylabel('Counts')
    plt.title(f'State-Action Pair Counts for Configuration {configuration + 1}')
    plt.show()

# Plot the total number of unique state-action pairs as a bar chart across all configurations
configurations = [str(i + 1) for i in range(len(epsilon_values))]
plt.figure(figsize=(12, 8))
plt.bar(configurations, all_total_unique_pairs)
plt.xlabel('Configurations')
plt.ylabel('Number of Unique State-Action Pairs')
plt.title('Total Unique State-Action Pairs Across Configurations')
plt.show()


# Function to build and display the Blackjack Strategy table
def build_blackjack_strategy_table(Q_values):
    # Define ranks and suits
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['Spades', 'Clubs', 'Hearts', 'Diamonds']
    
    # Initialize the strategy table
    strategy_table = {}
    
    # Iterate over player's sum and dealer's visible card
    for player_sum in range(20, 11, -1):
        strategy_table[player_sum] = {}
        for dealer_card in ranks[1:]:  # Exclude Ace for the dealer's card
            state = (player_sum, dealer_card)
            # Find the action with the highest Q-value in this state
            best_action = max(Q_values[state], key=Q_values[state].get)
            # Populate the cell of the strategy table
            strategy_table[player_sum][dealer_card] = 'H' if best_action == Action.HIT else 'S'
    
    return strategy_table

# Generate and display Blackjack Strategy tables for each configuration
for configuration in range(len(epsilon_values)):
    print(f"Blackjack Strategy Table for Configuration {configuration + 1}:")
    
    # Generate the strategy table for when the player is using an Ace as 11
    strategy_table_with_ace = build_blackjack_strategy_table(all_state_action_counts[configuration])
    print("Player's Ace as 11:")
    print("Dealer's Card |", " | ".join(strategy_table_with_ace[20].keys()))
    print("-" * (15 + 4 * len(strategy_table_with_ace[20])))
    for player_sum in range(20, 11, -1):
        print(f"Player's Sum {player_sum} |", " | ".join(strategy_table_with_ace[player_sum].values()))
    print()
    
    # Generate the strategy table for when the player is not using an Ace as 11
    strategy_table_without_ace = build_blackjack_strategy_table(all_state_action_counts[configuration])
    print("Player's Ace as 1:")
    print("Dealer's Card |", " | ".join(strategy_table_without_ace[20].keys()))
    print("-" * (15 + 4 * len(strategy_table_without_ace[20])))
    for player_sum in range(20, 11, -1):
        print(f"Player's Sum {player_sum} |", " | ".join(strategy_table_without_ace[player_sum].values()))
    print()
