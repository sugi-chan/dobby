from random import randint

import numpy as np
from keras.layers.core import Activation, Dense
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.models import load_model


def use_predicted_probability(label_dict, pred_key):
    return label_dict[pred_key]


def random_outfit(label_dict):
    outfit_to_play = randint(0, (len(label_dict)-1))
    return outfit_to_play


class OutfitBot():
    def __init__(self, type=None, label_dict=None):
        super().__init__()
        self.outfit_score = 0
        self.type = type
        self.label_dict = label_dict

    def get_action(self, state=None):
        return self.label_dict[randint(0, (len(self.label_dict)-1))]


class DQNLearner(OutfitBot):
    def __init__(self, label_dict):
        super().__init__()
        self._learning = True
        self._learning_rate = .001
        self._discount = .1
        self._epsilon = .7
        self.label_dict = label_dict
        self.len_vectorizer_matrix = 421
        print(self.label_dict)
        # Create Model
        # input. opp  last  played, player last played, opp_score,player_score, current_fishery_count, 
        model = Sequential()

        model.add(Dense(256, init='glorot_normal', activation='relu',
                        input_dim=self.len_vectorizer_matrix))

        model.add(Dense(256, init='glorot_normal', activation='relu'))
        model.add(Dense(128, init='glorot_normal', activation='relu'))
        #model.add(Dense(256, init='glorot_normal', activation='relu'))
        #output in this case should be a 60 way classification
        #representing the 60 ways you can choose 3 cards out of 5

        model.add(Dense(len(self.label_dict),
                  init='glorot_normal',
                  activation='linear'))

        opt = RMSprop()
        model.compile(loss='mse', optimizer=opt)

        self._model = model

    def get_action(self, state):

        game_state_array = np.reshape(np.asarray(state), (1, self.len_vectorizer_matrix))

        preds = self._model.predict(game_state_array, batch_size=1)

        predicted_classs = np.argmax(preds)

        if np.random.uniform(0, 1) < self._epsilon:
            action = use_predicted_probability(self.label_dict, predicted_classs)

        else:
            #if above the epsilon value, then choose one of 60
            predicted_classs = random_outfit(self.label_dict)
            action = use_predicted_probability(self.label_dict,predicted_classs)

        self._last_state = game_state_array
        self._last_action = predicted_classs
        self._last_target = preds

        return action

    def update(self, state, reward):
        '''
        reward:
                reward genearted from the game envionment
            state:
                game state
            new:
                discounted model outputs. This gets combined with with the game environment rewards
        '''
        if self._learning:
            outfit_state_array = np.reshape(np.asarray(state), (1, self.len_vectorizer_matrix))
            preds = self._model.predict([outfit_state_array], batch_size=1)
            maxQ = np.amax(preds)
            new = self._discount * maxQ

            combined_reward = reward + new

            self._last_target[0][self._last_action] = combined_reward

            #print(self.label_dict[self._last_action],self._last_action,reward,new,combined_reward,self._last_target[0])
            self._model.fit(self._last_state, self._last_target, batch_size=1, epochs=1, verbose=0)

    def save_rl_model(self, name_model):
        self._model.save(str(name_model)+'.h5')
