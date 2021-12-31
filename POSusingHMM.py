#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import division
import pandas as pd
from _collections import defaultdict
import json
import re
import sys
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# ## Task 1: Vocabulary Creation

# In[31]:


def readTrainFile(filename):
    df = pd.read_csv(filename, sep = "\t", names=["index","word", "tag"])
    return df


# In[32]:


data = readTrainFile('./data/train')
data.head()


# In[33]:


def get_freq(df):
    freq_data = df.groupby(['word']).size().reset_index(name='frequency').sort_values(['frequency'], ascending=False)
    freq_data = freq_data.reset_index(drop=True)
    return freq_data


# In[34]:


freq = get_freq(data)
freq.head()


# In[66]:


def replace_word(df):
    threshold = 2.0
    df.loc[(df['frequency'] <= threshold), 'word'] = '<unk>'
    return df


# In[67]:


replaced_df = replace_word(freq)
replaced_df.tail()


# In[68]:


def create_vocab(df):
    
    grouped_df = df.groupby(['word'])
    final = grouped_df.sum()
    final = final.sort_values(['frequency'], ascending=False).reset_index()
    
    data1 = final[final.word=='<unk>']
    data2 = final[final.word !='<unk>']
    
    final_df = pd.concat([data1, data2])
    
    final_df = final_df.reset_index()
    
    final_df.drop('index', axis=1, inplace=True)
    final_df['index'] = final_df.index
    final_df = final_df[['word', 'index', 'frequency']]
    
    return final_df


# In[69]:

vocab_df = create_vocab(replaced_df)
print(vocab_df.shape)
vocab_df.tail()

# In[70]:

vocab_df.head()

# In[71]:

def create_text_file(df, filename):
    df.to_csv(filename, sep = '\t', header = None)

# In[72]:

create_text_file(vocab_df, './data/vocab.txt')

# In[73]:

print("Selected Threshold: ", 2.0)
print("Total size of Vocabulary: ", vocab_df.shape)
print("Total occurrences of the special token ‘< unk >’ after replacement: ", vocab_df.head())


# ## Task 2: Model Learning

# In[43]:


# function to add set data structure in JSON file

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


# In[44]:

# function to calculate accuracy of algorithms

def calculate_accuracy(actual_tags, predicted_tags):
    
    act_tags_list = []
    pred_tags_list = []
    
    for i in range(0, len(predicted_tags)):
        for j in range(len(predicted_tags[i])):
            act_tags_list.append(actual_tags[i][j])
            pred_tags_list.append(predicted_tags[i][j])
            
    return accuracy_score(act_tags_list, pred_tags_list)*100


# In[45]:

def process_word_for_num(w):
    if (re.match('[0-9]+[,.-\\]*[0-9]*', w) is not None) and (re.match('[0-9]+[,.-\\]*[0-9]*[-]*[a-zA-Z]', w) is None):
        return '<num>'
    return w


# In[46]:

# function to calculate transition prob

def create_transition_dict(transition, context, numberOfStates):
    
    transitions_dict = dict()
    transitions_dict_with_freq = dict()
    
    for key in transition:
        previous, tag = key[0], key[1]
        key_str = f'{previous},{tag}'
        transitions_dict[key_str] = (transition[key] + 1) / (context[previous] + numberOfStates)
        transitions_dict_with_freq[key_str] = str(transition[key]) + ' ' + str(transitions_dict[key_str])
        
    return transitions_dict, transitions_dict_with_freq


# In[47]:

# function to calculate emission prob

def create_emission_dict(emmission, context):
    emmission_dict = dict()
    for key in emmission:
        tag, word = key[0], key[1]
        key_str = f'{tag},{word}'
        emmission_dict[key_str] = emmission[key] / context[tag]
        
    return emmission_dict


# In[48]:

# function to learn hmm model
def learn_hmm_model(train_file, output_file, extra_data_file):
    emmission = defaultdict(int)
    transition = defaultdict(int)
    context = defaultdict(int)
    word_tags_dict = dict()
    states = set()

    output_file = open(output_file, 'w')
    extra_data_file = open(extra_data_file, 'w')

    with open(train_file) as f:
        sentences = f.read().split('\n\n')
        for sentence in sentences:
            previous = '<s>'
            context[previous] += 1

            wordtags = sentence.strip().split('\n')

            for wordtag in wordtags:
                idx, word, tag = wordtag.split('\t')
                word = process_word_for_num(word)

                transition[(previous, tag)] += 1
                context[tag] += 1
                emmission[(tag, word)] += 1
                previous = tag

                if word not in word_tags_dict:
                    word_tags_dict[word] = set()

                word_tags_dict[word].add(tag)
                states.add(tag)
            transition[(previous, '</s>')] += 1
            
    numberOfStates = len(states)

    transitions_dict, transitions_dict_with_freq = create_transition_dict(transition, context, numberOfStates)
    emmission_dict = create_emission_dict(emmission, context)
   
    hmm_final_dict = {}
    hmm_final_dict['transition'] = transitions_dict
    hmm_final_dict['emission'] = emmission_dict
   

    hmm_json = json.dump(hmm_final_dict, output_file, indent=4)

    extra_data_dict = {}
    extra_data_dict['transition'] = transitions_dict_with_freq
    extra_data_dict['emission'] = emmission_dict
    extra_data_dict['context'] = context
    extra_data_dict['word_tags_dict'] = word_tags_dict
    extra_data_dict['states'] = states
    json.dump(extra_data_dict, extra_data_file, default=set_default)
    print("-----HMM Model parameters-----")
    print("No. of transition parameters: ", len(transitions_dict))
    print("No. of emmission parameters: ", len(emmission_dict))


learn_hmm_model('./data/train', './data/hmm.json', './data/extra_data.json')

# # Decoding

# In[50]:

def run_for_test_file(wordtags):
    
    test_words = []
    tags = []
    words = []
    
    for word in wordtags:
        idx, word = word.split('\t')
        words.append(word)
        word = process_word_for_num(word)
        test_words.append(word)
    return words, test_words


# In[51]:

def run_for_dev_file(wordtags):
    
    test_words = []
    tags = []
    words = []
    
    for word in wordtags:
        idx, word, tag = word.split('\t')
        words.append(word)
        word = process_word_for_num(word)
        test_words.append(word)
        tags.append(tag)
    return words, test_words, tags


# In[52]:

def write_output_file(output, tags_list, words, predicted_tags):
    
    for i in range(0, len(tags_list)):
        word, actual_tag = tags_list[i].split("/", 1)
        output.write(str(i+1)+'\t'+str(words[i])+'\t'+str(predicted_tags[i])+'\n')
    output.write('\n')
    


# In[53]:

def decoding_with_hmm(input_file, output_file, extra_data_file, algorithm_type, test_mode):
    
    emmission = defaultdict(int)
    transition = defaultdict(int)
    start = defaultdict(int)
    totalStart = 0

    extra_data = open(extra_data_file)

    data = json.load(extra_data)
    for key, val in data['transition'].items():
        tmp_val = val.split(' ')
        tmp_val[0] = int(tmp_val[0])
        transition[key] = float(tmp_val[1])

        if key.startswith('<s>'):
            temp = key.split(',')
            start[str(temp[1].strip())] += tmp_val[0]
            totalStart += tmp_val[0]

    emmission = data['emission']
    context = data['context']
    states = set(data['states'])
    word_tags_dict = data['word_tags_dict']

    for i in states:
        start[i] = start[i] * totalStart

    output = open(output_file, 'w')
    actual_tags = []
    predicted_tags = []
    
    with open(input_file) as f:
        sentences = f.read().split('\n\n')

        for line in sentences:
            wordtags = line.strip().split('\n')
            
            if test_mode == 'test':
                words, test_words = run_for_test_file(wordtags)                 
            elif test_mode == 'dev':
                words, test_words, tags = run_for_dev_file(wordtags)
                actual_tags.append(tags)
                
            if algorithm_type == 0:
                listOfTags, predicted_sentence_tags = greedy_decoding(test_words, states,  start, transition, emmission, word_tags_dict)
            elif algorithm_type == 1:
                listOfTags, predicted_sentence_tags = viterbi_decoding(test_words, states,  start, transition, emmission, word_tags_dict)
            
            predicted_tags.append(predicted_sentence_tags)
            write_output_file(output, listOfTags, words, predicted_sentence_tags)
            
    if test_mode == 'dev':        
        print('Accuracy: ', calculate_accuracy(actual_tags, predicted_tags))
        


# ## Task 3

# In[54]:

def greedy_decoding(test_words, states, start, transition, emmission, word_tags_dict):
    
    max_track_dict = []
    present_states = states
    max_val = -1
    tag_with_max = ''
    
    if test_words[0] in word_tags_dict:
        
        present_states = word_tags_dict[test_words[0]]
        
        for i in present_states:
            if i in states and start[i] > 0 and f'{i},{test_words[0].strip()}' in emmission:
                val = start[i]*emmission[f'{i},{test_words[0].strip()}']
                if max_val <= val:
                    max_val = val
                    tag_with_max = i
    else:
        
        for i in states:
            if i in states and start[i] > 0:
                if max_val <= start[i]:
                    max_val = start[i]
                    tag_with_max = i

    max_track_dict.append((max_val, tag_with_max))

    if '<s>' in states:
        states.remove('<s>')

    for t in range(1, len(test_words)):
        present_states = states
        max_val = -1
        tag_with_max = ''
        
        if test_words[t] in word_tags_dict:
            present_states = word_tags_dict[test_words[t]]
            for y in present_states:
                prev_word = max_track_dict[t-1]  
                probvalv = prev_word[0] * transition[f'{prev_word[1]},{y}'] * emmission[f'{y},{test_words[t].strip()}']
                if max_val <= probvalv:
                    max_val = probvalv
                    tag_with_max = y

        else:
            for y in present_states:
                prev_word = max_track_dict[t-1]  
                probvalv = prev_word[0] * transition[f'{prev_word[1]},{y}']
                if max_val < probvalv:
                    max_val = probvalv
                    tag_with_max = y

        max_track_dict.append((max_val, tag_with_max))

    listOfTags = []
    predicted_tags = []

    for word, tag in zip(test_words, max_track_dict):

        listOfTags.append(word+'/'+tag[1])
        predicted_tags.append(tag[1])
    
    return listOfTags, predicted_tags


# In[55]:

print("-----Greedy Decoding Algorithm-----")
decoding_with_hmm('./data/dev', './data/greedy_dev_data.out', './data/extra_data.json', 0, 'dev')


# In[56]:


decoding_with_hmm('./data/test', './data/greedy.out', './data/extra_data.json', 0, 'test')


# ## Task 4

# In[57]:


def viterbi_decoding(test_words, states, start, transition, emmission, word_tags_dict):
    
    dp = [defaultdict(int)]
    backtrack_tags = [defaultdict(int)]
    present_states = states
    if test_words[0] in word_tags_dict:
        present_states = word_tags_dict[test_words[0]]
        for i in present_states:
           
            if i in states and start[i] > 0 and f'{i},{test_words[0].strip()}' in emmission:
                dp[0][i] = start[i]*emmission[f'{i},{test_words[0].strip()}']

    else:
        for i in states:
            if i in states and start[i] > 0:
                dp[0][i] = start[i]

    if '<s>' in states:
        states.remove('<s>')
    for t in range(1, len(test_words)):
        dp.append(defaultdict(int))
        backtrack_tags.append(defaultdict(int))
        present_states = states
        if test_words[t] in word_tags_dict:
            present_states = word_tags_dict[test_words[t]]

            for y in present_states:
                maxvalv = -1
                for y0 in states:
                    if((y0 in dp[t-1]) and dp[t-1][y0] > 0 and (f'{y},{test_words[t].strip()}' in emmission)):
                        probvalv = dp[t-1][y0] * transition[f'{y0},{y}'] * emmission[f'{y},{test_words[t].strip()}']
                        if maxvalv <= probvalv:
                            maxvalv = probvalv
                            backtrack_tags[t][y] = y0
                dp[t][y] = maxvalv

        else:

            for y in states:
                maxvalv = -1
                for y0 in states:
                    if((y0 in dp[t-1]) and dp[t-1][y0] > 0):
                        probvalv = dp[t-1][y0]*transition[f'{y0},{y}']
                        if maxvalv <= probvalv:
                            maxvalv = probvalv
                            backtrack_tags[t][y] = y0
                dp[t][y] = maxvalv

    listOfTags = []
    predicted_tags = []
    tag = ''
    maxval = -1
    present_states = states
    if test_words[len(test_words) - 1] in word_tags_dict:
        present_states = word_tags_dict[test_words[len(test_words) - 1]]

    for i in present_states:
        if maxval <= dp[len(test_words) - 1][i]:
            maxval = dp[len(test_words) - 1][i]
            tag = i

    listOfTags.append(test_words[len(test_words) - 1]+'/'+tag)
    predicted_tags.append(tag)
    for t in range(len(test_words) - 2, -1, -1):
        if tag not in backtrack_tags[t+1]:
            for i in states:
                if i in backtrack_tags[t+1]:
                    tag = backtrack_tags[t+1][i]
                    break
        else:
            tag = backtrack_tags[t+1][tag]
        listOfTags.append(test_words[t]+'/'+tag)
        predicted_tags.append(tag)

    predicted_tags.reverse()
    listOfTags.reverse()
    return listOfTags, predicted_tags


# In[58]:

print("-----Viterbi Decoding Algorithm-----")
decoding_with_hmm('./data/dev', './data/viterbi_dev_data.out', './data/extra_data.json', 1, 'dev')


# In[59]:


decoding_with_hmm('./data/test', './data/viterbi.out', './data/extra_data.json', 1, 'test')


# In[ ]:




