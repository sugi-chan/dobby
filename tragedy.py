from random import randint
from netlearner import random_outfit


def convert_agent_output_to_judge_input(input_str):
    broken_out = input_str.replace('_',' ')
    input_str = broken_out +' '+input_str
    return input_str


class OutfitEnv():
    def __init__(self,
                 num_learning_rounds=None,
                 learner=None,
                 report_every=1000,
                 vectorizer=None):
        super().__init__()
        self._num_learning_rounds = num_learning_rounds
        self._report_every = report_every
        self.player = learner #change to learner later
        self.game = 1
        self.evaluation = False
        self.outfit_records = []
        self.vectorizer = vectorizer

    def play_game(self, eval=False):

        p1, initial_item, outfit_score = self.reset_game()
        #opp  last  played, player last played, opp_score,player_score,turn#, current_fishery_count, 
        state = initial_item
        #print(state, outfit_score)
        pre_outfit_score = outfit_score
        #every "game" is going to be some number of rounds where it picks an item at every round
        # for now setting it to 4
        if eval== True:
            print('eval round 1 state 1: ', state)
        for turn in range(4):
            vectorized_state = self.vectorizer.transform([convert_agent_output_to_judge_input(state)])
            if eval == True:
                print('turn {}: '.format(turn), state)
                #print('vectorized',vectorized_state)
            vectorized_state = list(vectorized_state.toarray()[0])

            next_clothing_choice = p1.get_action(vectorized_state)
            state_after_pred = state + ' ' + next_clothing_choice
            
            if 'bottom' in state_after_pred:
                round_outfit_score = state_after_pred.count('bottom_')*10
            else:
                round_outfit_score = 1

            #round_outfit_score = 1 # get outfit score from model and rules
            round_score_diff = round_outfit_score - pre_outfit_score
            outfit_score +=round_score_diff
            if eval == True:
                print('turn {} after action: '.format(turn), state_after_pred)
                print('score after action', round_outfit_score)
                print('difference from pre round',round_score_diff,round_outfit_score,pre_outfit_score)

            pre_outfit_score = round_outfit_score

            
            if eval == True:
                print('round outfit score',round_outfit_score,pre_outfit_score)
                print(state,round_score_diff)

            state = state_after_pred

            p1.update(vectorized_state, round_score_diff) #state + reward from the environment


            if self.evaluation == True:
                print('total outfit score :', outfit_score)

        if self.evaluation == False:
            self.game += 1
            self.outfit_records.append(outfit_score)
            self.report()

        if self.game == self._num_learning_rounds:
            print("Turning off learning!")
            self.player._learning = False
            self.win = 0
            self.loss = 0

    def reset_game(self):
        '''
        steps to reset:
        1) get random item out of the product catalog can select a thing out of the label dict

        ## random action refactor
        def random_outfit(tax_att_dict):
            outfit_to_play = randint(0,(len(tax_att_dict)-1))
            return outfit_to_play

        score set to 0
        '''
        p1 = self.player
        outfit_score = 0 # outfitt score should be sent through model/rules for a 1 outfit item 
        initial_item = self.player.label_dict[random_outfit(self.player.label_dict)]

        return p1,initial_item,outfit_score
    
    def report(self):
        #turned off for plotting 9/18
        if self.game % self._num_learning_rounds == 0:

            avg_outfit_score =  sum(self.outfit_records) / float(len(self.outfit_records))
            self.outfit_records =[]
            print('##############################################')
            print('#                 Final Score                #')
            print('##############################################')
            print('')
            print(str(self.game))
            print('')
            print('average outfit score: ',avg_outfit_score)
            print('##############################################')
            
            #turning off this section for testing
            self.evaluation = True
            self.player._epsilon = 1.0
            print('#################### G1 ######################')
            self.play_game(eval = True)
            print('#################### G2 ######################')
            self.play_game(eval = True)
            print('#################### G3 ######################')
            self.play_game(eval = True)
            self.player._epsilon = .9
            self.evaluation = False
            
            #self.player.save_rl_model('models/{}_iteration_{}'.format(rl_model_name,self.game))
            
        elif self.game % self._report_every == 0:
            avg_outfit_score =  sum(self.outfit_records) / float(len(self.outfit_records))
            self.outfit_records =[]
            print('##############################################')
            print('#                Updated Score               #')
            print('##############################################')
            print('')
            print(str(self.game))
            print('')
            print('average outfit score: ',avg_outfit_score)
            print('##############################################')
            
            #turning off this section for testing
            self.evaluation = True
            self.player._epsilon = 1.0
            print('#################### G1 ######################')
            self.play_game(eval = True)
            print('#################### G2 ######################')
            self.play_game(eval = True)
            print('#################### G3 ######################')
            self.play_game(eval = True)
            self.player._epsilon = .9
            self.evaluation = False
            
            #self.player.save_rl_model('models/{}_iteration_{}'.format(rl_model_name,self.game))
   
#game1 = Tragedy()
#game1.play_game()