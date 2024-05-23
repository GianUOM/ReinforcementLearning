import random
import numpy as np
import math

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
        self.dealer_cards = [self.deck.draw_card(), self.deck.draw_card()]

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

class Sarsa:
    def __init__(self, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma  
        self.Q = {}  # Q-table

    def get_action(self, state):
        if state not in self.Q:
            self.Q[state] = {Action.HIT: 0, Action.STAND: 0}
        if random.random() < self.epsilon:
            return random.choice([Action.HIT, Action.STAND])
        else:
            return max(self.Q[state], key=self.Q[state].get)

    def update_Q(self, state, action, reward, next_state, next_action):
        if state not in self.Q:
            self.Q[state] = {Action.HIT: 0, Action.STAND: 0}
        if next_state not in self.Q:
            self.Q[next_state] = {Action.HIT: 0, Action.STAND: 0}
        td_target = reward + self.gamma * self.Q[next_state][next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

        # Usage example with statistics output
# Usage example with statistics output
episode = 1
epsilons = [0.1, 1/episode, math.exp(-episode/1000), math.exp(-episode/10000)]

configuration = 1
for epsilon in epsilons:
    if callable(epsilon):  
        epsilon_name = epsilon.__name__
        sarsa_agent = Sarsa(epsilon=epsilon)
    else:
        epsilon_name = str(epsilon)
        sarsa_agent = Sarsa(epsilon=epsilon)
    
    
    print(f"Running configuration {configuration}")
    configuration+=1

    episode_rewards = []
    win_count = 0
    loss_count = 0
    draw_count = 0
    state_action_counts = {}
    unique_state_action_pairs = set()

    for episode in range(100000):
        round = BlackjackRound()
        round.start()
        state = round.get_state()
        action = sarsa_agent.get_action(state)
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
            next_action = sarsa_agent.get_action(next_state)
            
            sarsa_agent.update_Q(state, action, reward, next_state, next_action)
            
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
        sarsa_agent.update_Q(state, action, reward, next_state, next_action)

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
            win_count = 0
            loss_count = 0
            draw_count = 0

    # Output final statistics
    final_avg_reward = np.mean(episode_rewards)
    print(f"Final Average Reward: {final_avg_reward}")
    print(f"Number of unique state-action pairs explored: {len(unique_state_action_pairs)}")
    print("State-Action Pair Counts:")
    for pair, count in state_action_counts.items():
        print(f"State: {pair[0]}, Action: {pair[1]}, Count: {count}")
    print("Estimated Q values:")
    for pair in unique_state_action_pairs:
        print(f"State: {pair[0]}, Action: {pair[1]}, Q-value: {sarsa_agent.Q[pair[0]][pair[1]]}")
