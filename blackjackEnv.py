import numpy as np
import random

# Blackjack Environment
class BlackjackEnvironment:
    def __init__(self):
        self.deck = self.initialize_deck()
        self.player_hand = []
        self.dealer_hand = []
        self.reset()
    
    def initialize_deck(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        deck = [{'rank': rank, 'value': self.card_value(rank)} for rank in ranks for _ in range(4)]
        random.shuffle(deck)
        return deck
    
    def card_value(self, rank):
        if rank in ['J', 'Q', 'K']:
            return 10
        elif rank == 'A':
            return 11
        else:
            return int(rank)
    
    def deal_card(self):
        if not self.deck:
            self.deck = self.initialize_deck()
        return self.deck.pop()
    
    def calculate_hand_value(self, hand):
        value = sum(card['value'] for card in hand)
        ace_count = sum(1 for card in hand if card['rank'] == 'A')
        while value > 21 and ace_count:
            value -= 10
            ace_count -= 1
        return value
    
    def reset(self):
        self.player_hand = [self.deal_card(), self.deal_card()]
        self.dealer_hand = [self.deal_card(), self.deal_card()]
        return self.get_state()
    
    def get_state(self):
        player_value = self.calculate_hand_value(self.player_hand)
        dealer_upcard_value = self.dealer_hand[0]['value']
        return np.array([[player_value, dealer_upcard_value]])
    
    def step(self, action):
        done = False
        reward = 0
        loss_reason = ""
        
        print("Player's hand before action:", self.player_hand)
        print("Dealer's hand before action:", self.dealer_hand)
        
        if action == 0:  # Hit
            print("Action: Hit")
            self.player_hand.append(self.deal_card())
            player_value = self.calculate_hand_value(self.player_hand)
            print("Player's hand after hit:", self.player_hand)
            if player_value > 21:
                print("Player busts")
                print("-----")
                return self.get_state(), -1, True, "Player busts"
        
        # If action is not hit, it must be stand
        print("Action: Stand")
        player_value = self.calculate_hand_value(self.player_hand)
        dealer_value = self.calculate_hand_value(self.dealer_hand)
        while dealer_value < 17:
            self.dealer_hand.append(self.deal_card())
            dealer_value = self.calculate_hand_value(self.dealer_hand)
        print("Dealer's hand after drawing:", self.dealer_hand)
        
        if dealer_value > 21:
            print("Dealer busts")
            print("Player Value:", player_value)
            print("Dealer Value:", dealer_value)
            print("Reward: 1")
            print("Done: True")
            print("Loss Reason: Dealer busts")
            print("-----")
            return self.get_state(), 1, True, "Dealer busts"
        elif player_value > dealer_value:
            print("Player wins")
            print("Player Value:", player_value)
            print("Dealer Value:", dealer_value)
            print("Reward: 1")
            print("Done: True")
            print("Loss Reason: Player has a stronger hand")
            print("-----")
            return self.get_state(), 1, True, "Player has a stronger hand"
        elif player_value < dealer_value:
            print("Dealer wins")
            print("Player Value:", player_value)
            print("Dealer Value:", dealer_value)
            print("Reward: -1")
            print("Done: True")
            print("Loss Reason: Dealer has a stronger hand")
            print("-----")
            return self.get_state(), -1, True, "Dealer has a stronger hand"
        else:  # player_value == dealer_value
            print("It's a draw")
            print("Player Value:", player_value)
            print("Dealer Value:", dealer_value)
            print("Reward: 0")
            print("Done: True")
            print("Loss Reason: It's a draw")
            print("-----")
            return self.get_state(), 0, True, "It's a draw"





