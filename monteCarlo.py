import random
import numpy as np
import matplotlib.pyplot as plt
from Sarsanew import *
from Qlearningnew import *

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

def generateState(player_sum, dealer_card, has_ace):
    return (player_sum, dealer_card, has_ace)

def chooseAction(q_values, state, epsilon):
    if state not in q_values or not q_values[state]:
        return np.random.choice(['HIT', 'STAND'])
    if random.random() < epsilon:
        return np.random.choice(['HIT', 'STAND'])
    else:
        return max(q_values[state], key=q_values[state].get)

def monteCarloUsingExploringStarts(numberOfEpisodes):
    q_values = {}
    for episode in range(1, numberOfEpisodes + 1):
        round = BlackjackRound()
        round.start()
        state = generateState(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
        episode_actions = []
        episode_states = []
        episode_rewards = []

        if round.get_sum(round.player_cards) in range(12, 21):
            action = np.random.choice(['HIT', 'STAND'])
        else:
            action = chooseAction(q_values, state, 1)  

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

            state = generateState(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
            episode_states.append(state)

            epsilon = 1 / episode
            if round.get_sum(round.player_cards) in range(12, 21):
                action = chooseAction(q_values, state, epsilon)
            else:
                action = chooseAction(q_values, state, 1)  
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

def monteCarloUsingNonExploringStarts(numberOfEpisodes, alpha, epsilon_func):
    q_values = {}
    for episode in range(numberOfEpisodes):
        round = BlackjackRound()
        round.start()
        state = generateState(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
        episode_actions = []
        episode_rewards = []

        action = chooseAction(q_values, state, epsilon_func(episode + 1))  
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

            next_state = generateState(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
            state = next_state

            action = chooseAction(q_values, state, epsilon_func(episode + 1))  
            episode_actions.append(action)

        for i in range(len(episode_actions)):
            stateAction = (state, episode_actions[i])
            G = sum(episode_rewards[i:])
            if state not in q_values:
                q_values[state] = {}
            if episode_actions[i] not in q_values[state]:
                q_values[state][episode_actions[i]] = 0
            q_values[state][episode_actions[i]] += alpha * (G - q_values[state][episode_actions[i]])

    return q_values

def extractingInfoThroughEpisodes(agent_function, numberOfEpisodes, alpha, epsilon_func):
    q_values = {}
    episodeResults = {'Win': [], 'Loss': [], 'Draw': []}
    uniqueStateActionPairs = set()
    actionCounts = {}
    winsPerEpisode = []
    lossesPerEpisode = []
    drawsPerEpisode = []

    for episode in range(1, numberOfEpisodes + 1):
        round = BlackjackRound()
        round.start()
        state = generateState(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
        episode_actions = []
        episode_rewards = []

        action = chooseAction(q_values, state, epsilon_func(episode))  
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

            next_state = generateState(round.get_sum(round.player_cards), round.dealer_cards[0].rank, 'A' in [card.rank for card in round.player_cards])
            state = next_state

            action = chooseAction(q_values, state, epsilon_func(episode))  
            episode_actions.append(action)

        for i in range(len(episode_actions)):
            stateAction = (state, episode_actions[i])
            G = sum(episode_rewards[i:])
            if state not in q_values:
                q_values[state] = {}
            if episode_actions[i] not in q_values[state]:
                q_values[state][episode_actions[i]] = 0
            q_values[state][episode_actions[i]] += alpha * (G - q_values[state][episode_actions[i]])

        episodeResult = round.get_outcome()
        episodeResults[episodeResult].append(1)

        if episodeResult == 'Win':
            winsPerEpisode.append(1)
            lossesPerEpisode.append(0)
            drawsPerEpisode.append(0)
        elif episodeResult == 'Loss':
            winsPerEpisode.append(0)
            lossesPerEpisode.append(1)
            drawsPerEpisode.append(0)
        else:
            winsPerEpisode.append(0)
            lossesPerEpisode.append(0)
            drawsPerEpisode.append(1)

        uniqueStateActionPairs.add(stateAction)

        if (state, action) in actionCounts:
            actionCounts[(state, action)] += 1
        else:
            actionCounts[(state, action)] = 1

        if episode % 1000 == 0:
            print(f"Episodes {episode-999}-{episode}: Wins - {sum(episodeResults['Win'])}, Losses - {sum(episodeResults['Loss'])}, Draws - {sum(episodeResults['Draw'])}")
            episodeResults = {'Win': [], 'Loss': [], 'Draw': []}

    return winsPerEpisode, lossesPerEpisode, drawsPerEpisode, sum(episodeResults['Win']), sum(episodeResults['Loss']), sum(episodeResults['Draw']), len(uniqueStateActionPairs), actionCounts, q_values



numberOfEpisodes = 100000
alpha = 0.1
winsPerEpisode_explore, lossesPerEpisode_explore, drawsPerEpisode_explore, winsExploring, lossesExploring, drawsExploring, uniquePairsExplored, actionCounts_explore, qValuesExploring = extractingInfoThroughEpisodes(monteCarloUsingExploringStarts, numberOfEpisodes, alpha, lambda k: 1 / k)

numberOfEpisodes = 100000
alpha = 0.1
winsPerEpisode1, lossesPerEpisode1, drawsPerEpisode1, winsNonExploring1, lossesNonExploring1, drawsNonExploring1, uniquePairsNonExploring1, actionCountsNonExploring1, qValuesNonExploring1 = extractingInfoThroughEpisodes(monteCarloUsingNonExploringStarts, numberOfEpisodes, alpha, lambda k: 1 / k)

numberOfEpisodes = 100000
alpha = 0.1
winsPerEpisode2, lossesPerEpisode2, drawsPerEpisode2, winsNonExploring2, lossesNonExploring2, drawsNonExploring2, uniquePairsNonExploring2, actionCountsNonExploring2, qValuesNonExploring2 = extractingInfoThroughEpisodes(monteCarloUsingNonExploringStarts, numberOfEpisodes, alpha, lambda k: np.exp(-k / 1000))

numberOfEpisodes = 100000
alpha = 0.1
winsPerEpisode3, lossesPerEpisode3, drawsPerEpisode3, winsNonExploring3, lossesNonExploring3, drawsNonExploring3, uniquePairsNonExploring3, actionCountsNonExploring3, qValuesNonExploring3 = extractingInfoThroughEpisodes(monteCarloUsingNonExploringStarts, numberOfEpisodes, alpha, lambda k: np.exp(-k / 10000))

print('-----------------------------------------------------------------------')
print("\nResults for Exploring Starts:\n")
print("Wins: ", winsExploring)
print("Losses: ", lossesExploring)
print("Draws: ", drawsExploring)
print("Unique state-action pairs explored: ", uniquePairsExplored)
print("\nCounts of state-action pair selections:")
for stateAction, count in actionCounts_explore.items():
    print(f"State-Action Pair: {stateAction}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in qValuesExploring.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")

print('-----------------------------------------------------------------------')
print("\nResults for Non-exploring Starts with 𝜖 = 1/k:\n")
print("Wins: ", winsNonExploring1)
print("Losses: ", lossesNonExploring1)
print("Draws: ", drawsNonExploring1)
print("Unique state-action pairs explored: ", uniquePairsNonExploring1)
print("\nCounts of state-action pair selections:")
for stateAction, count in actionCountsNonExploring1.items():
    print(f"State-Action Pair: {stateAction}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in qValuesNonExploring1.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")

print('-----------------------------------------------------------------------')
print("\nResults for Non-exploring Starts with 𝜖 = e^(-k/1000):\n")
print("Wins: ", winsNonExploring2)
print("Losses: ", lossesNonExploring2)
print("Draws: ", drawsNonExploring2)
print("Unique state-action pairs explored: ", uniquePairsNonExploring2)
print("\nCounts of state-action pair selections:")
for stateAction, count in actionCountsNonExploring2.items():
    print(f"State-Action Pair: {stateAction}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in qValuesNonExploring2.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")

print('-----------------------------------------------------------------------')
print("\nResults for Non-exploring Starts with 𝜖 = e^(-k/10000):\n")
print("Wins: ", winsNonExploring3)
print("Losses: ", lossesNonExploring3)
print("Draws: ", drawsNonExploring3)
print("Unique state-action pairs explored: ", uniquePairsNonExploring3)
print("\nCounts of state-action pair selections:")
for stateAction, count in actionCountsNonExploring3.items():
    print(f"State-Action Pair: {stateAction}, Count: {count}")
print("\nEstimated Q values:")
for state, actions in qValuesNonExploring3.items():
    for action, value in actions.items():
        print(f"State: {state}, Action: {action}, Q-value: {value}")

def plot_results(winsPerEpisode, lossesPerEpisode, drawsPerEpisode, numberOfEpisodes, title):
    interval = 1000
    winsInterval = []
    lossesInterval = []
    drawsInterval = []

    for i in range(0, numberOfEpisodes, interval):
        winsInterval.append(sum(winsPerEpisode[i:i+interval]))
        lossesInterval.append(sum(lossesPerEpisode[i:i+interval]))
        drawsInterval.append(sum(drawsPerEpisode[i:i+interval]))

    plt.figure(figsize=(10, 6))
    plt.plot(range(0, numberOfEpisodes, interval), winsInterval, label='Wins', marker='o', linestyle='-', color='blue')
    plt.plot(range(0, numberOfEpisodes, interval), lossesInterval, label='Losses', marker='o', linestyle='-', color='red')
    plt.plot(range(0, numberOfEpisodes, interval), drawsInterval, label='Draws', marker='o', linestyle='-', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Counts')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_actionCounts(actionCounts, title):
    validPlayerSums = list(range(12, 21))  
    validDealerCards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']

    filtered_actionCounts = {
        stateAction: count for stateAction, count in actionCounts.items()
        if stateAction[0][0] in validPlayerSums and stateAction[0][1] in validDealerCards
    }

    sorted_actionCounts = sorted(filtered_actionCounts.items(), key=lambda x: x[1], reverse=True)
    states_actions, counts = zip(*sorted_actionCounts)

    plt.figure(figsize=(30, 20))  
    plt.bar(range(len(states_actions)), counts)
    plt.xlabel('State-Action Pairs')
    plt.ylabel('Counts') 
    plt.title(title)
    plt.tight_layout()  
    plt.show()

def plot_uniqueStateActionPairs(configurations, unique_pairs_counts):
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

def analyze_last_10000_episodes(winsPerEpisode, lossesPerEpisode, drawsPerEpisode, title):
    total_wins_last_10000 = sum(winsPerEpisode[-10000:])
    total_losses_last_10000 = sum(lossesPerEpisode[-10000:])
    total_draws_last_10000 = sum(drawsPerEpisode[-10000:])
    print(f"{title} - Last 10000 Episodes: Wins = {total_wins_last_10000}, Losses = {total_losses_last_10000}, Draws = {total_draws_last_10000}")
    return total_wins_last_10000, total_losses_last_10000, total_draws_last_10000

def calculate_dealer_advantage(total_wins, total_losses):
    mean_wins = total_wins / 10000
    mean_losses = total_losses / 10000
    return (mean_losses - mean_wins) / (mean_losses + mean_wins)

def compare_dealer_advantage(configurations, advantages):
    min_advantage_index = advantages.index(min(advantages))
    min_advantage_config = configurations[min_advantage_index]

    
    plt.figure(figsize=(20, 10))
    bars = plt.bar(configurations, advantages, color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow'])

    plt.xlabel('Algorithm Configurations')
    plt.ylabel("Dealer's Advantage")
    plt.title('Dealer Advantage Across Different Algorithm Configurations')
    plt.grid(True)
    plt.xticks(rotation = 45, ha = 'center', fontsize = 10)
    plt.show()

    print(f"\nThe algorithm that minimizes the dealer advantage the most is: {min_advantage_config}")

plot_results(winsPerEpisode_explore, lossesPerEpisode_explore, drawsPerEpisode_explore, numberOfEpisodes, 'Exploring Starts')
plot_results(winsPerEpisode1, lossesPerEpisode1, drawsPerEpisode1, numberOfEpisodes, "Non-exploring Starts (1/k)")
plot_results(winsPerEpisode2, lossesPerEpisode2, drawsPerEpisode2, numberOfEpisodes, "Non-exploring Starts (e^(-k/1000))")
plot_results(winsPerEpisode1, lossesPerEpisode3, drawsPerEpisode3, numberOfEpisodes, "Non-exploring Starts (e^(-k/10000))")
plot_actionCounts(actionCounts_explore, 'Counts of State-Action Pairs: Exploring Starts')
plot_actionCounts(actionCountsNonExploring1, 'Counts of State-Action Pairs: Non-exploring Starts (1/k)')
plot_actionCounts(actionCountsNonExploring2, 'Counts of State-Action Pairs: Non-exploring Starts (e^(-k/1000))')
plot_actionCounts(actionCountsNonExploring3, 'Counts of State-Action Pairs: Non-exploring Starts (e^(-k/10000))')
configurations = ['Exploring Starts', 'Non-exploring Starts (1/k)', 'Non-exploring Starts (e^(-k/1000))', 'Non-exploring Starts (e^(-k/10000))']
unique_pairs_counts = [uniquePairsExplored, uniquePairsNonExploring1, uniquePairsNonExploring2, uniquePairsNonExploring3]
plot_uniqueStateActionPairs(configurations, unique_pairs_counts)

configs = [
    ("Exploring Starts", qValuesExploring),
    ("Non-exploring Starts with 𝜖 = 1/k", qValuesNonExploring1),
    ("Non-exploring Starts with 𝜖 = e^(-k/1000)", qValuesNonExploring2),
    ("Non-exploring Starts with 𝜖 = e^(-k/10000)", qValuesNonExploring3)
]

for config_name, q_values in configs:
    strategy_table_ace = build_strategy_table(q_values, has_ace=True)
    print_strategy_table(strategy_table_ace, f"Strategy Table with Ace as 11 - {config_name}")
    
    strategy_table_no_ace = build_strategy_table(q_values, has_ace=False)
    print_strategy_table(strategy_table_no_ace, f"Strategy Table without Ace as 11 - {config_name}")


# Calculate and analyze for each configuration
configurations = [
    "Exploring Starts",
    "Non-exploring Starts (1/k)",
    "Non-exploring Starts (e^(-k/1000))",
    "Non-exploring Starts (e^(-k/10000))",
    "SARSA (epsilon = 0.1)",
    "SARSA (epsilon = 1/k)",
    "SARSA (epsilon = e^(-k/1000))",
    "SARSA (epsilon = e^(-k/10000))",
    "Q-Learning (epsilon = 0.1)",
    "Q-Learning (epsilon = 1/k)",
    "Q-Learning (epsilon = e^(-k/1000))",
    "Q-Learning (epsilon = e^(-k/10000))"
]

results = [
    analyze_last_10000_episodes(winsPerEpisode_explore, lossesPerEpisode_explore, drawsPerEpisode_explore, 'Exploring Starts'),
    analyze_last_10000_episodes(winsPerEpisode1, lossesPerEpisode1, drawsPerEpisode1, 'Non-exploring Starts with 𝜖 = 1/k'),
    analyze_last_10000_episodes(winsPerEpisode2, lossesPerEpisode2, drawsPerEpisode2, 'Non-exploring Starts with 𝜖 = e^(-k/1000)'),
    analyze_last_10000_episodes(winsPerEpisode3, lossesPerEpisode3, drawsPerEpisode3, 'Non-exploring Starts with 𝜖 = e^(-k/10000)'),
    analyze_last_10000_episodes(wins_per_episode_config1, losses_per_episode_sansa1, draws_per_episode_sansa1, 'SARSA (epsilon = 0.1)'),
    analyze_last_10000_episodes(wins_per_episode_config2, losses_per_episode_sansa2, draws_per_episode_sansa2, 'SARSA (epsilon = 1/k)'),
    analyze_last_10000_episodes(wins_per_episode_config3, losses_per_episode_sansa3, draws_per_episode_sansa3, 'SARSA (epsilon = e^(-k/1000))'),
    analyze_last_10000_episodes(wins_per_episode_config4, losses_per_episode_sansa4, draws_per_episode_sansa4, 'SARSA (epsilon = e^(-k/10000))'),
    analyze_last_10000_episodes(wins_per_episode_config1, losses_per_episode_config1, draws_per_episode_config1, 'Q-Learning (epsilon = 0.1)'),
    analyze_last_10000_episodes(wins_per_episode_config2, losses_per_episode_config2, draws_per_episode_config2, 'Q-Learning (epsilon = 1/k)'),
    analyze_last_10000_episodes(wins_per_episode_config3, losses_per_episode_config3, draws_per_episode_config3, 'Q-Learning (epsilon = e^(-k/1000))'),
    analyze_last_10000_episodes(wins_per_episode_config4, losses_per_episode_config4, draws_per_episode_config4, 'Q-Learning (epsilon = e^(-k/10000))')
]

advantages = [calculate_dealer_advantage(total_wins, total_losses) for total_wins, total_losses, total_draws in results]

# Compare dealer advantage
compare_dealer_advantage(configurations, advantages)
