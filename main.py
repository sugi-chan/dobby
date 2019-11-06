from outfiting_env import convert_agent_output_to_judge_input, OutfitEnv
from netlearner import DQNLearner
import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def main():
    print('starting')
    dat2 = pd.read_csv('outfit_w_color_pattern.csv')

    dat2['combined_tax_att'] = ''
    for index, row in dat2.iterrows():
        tax_cat = row['taxonomy_cat']
        pattern = row['pattern_tonks']
        color = row['color_tonks']

        combined_cat = tax_cat+'_'+str(pattern)+'_'+str(color)
        combined_cat = combined_cat.replace('[', '')
        combined_cat = combined_cat.replace(']', '')
        combined_cat = combined_cat.replace("'", '')

        dat2.loc[index, 'combined_tax_att'] = combined_cat
    dat2['for_judge'] = ''
    for index, row in dat2.iterrows():
        tax_att = row['combined_tax_att']
        dat2.loc[index, 'for_judge'] = convert_agent_output_to_judge_input(tax_att)

    combined_text_list = list(dat2['for_judge'].unique())
    print('making label_dict for agent')
    tax_att_dict = {}
    index = 0
    #tax_att_dict[0] = 'skip'
    for i in combined_text_list:
        tax_att_dict[index] = i
        index += 1

    print('building vectorizer')
    corpus = dat2['for_judge'].tolist()

    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(corpus)
    print('num features: ', len(vectorizer.get_feature_names()),vectorizer.get_feature_names())

    num_learning_rounds = 20000
    game = OutfitEnv(num_learning_rounds=num_learning_rounds,
                     learner=DQNLearner(label_dict=tax_att_dict),
                     vectorizer=vectorizer)
    number_of_test_rounds = 1000
    for k in range(0, num_learning_rounds + number_of_test_rounds):
        game.play_game()

    #df = game.p.get_optimal_strategy()
    #print(df)
    #df.to_csv('optimal_policy.csv')


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))