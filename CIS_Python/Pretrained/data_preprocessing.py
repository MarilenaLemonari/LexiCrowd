import torch

# PARAMETERS
max_seq_len = 47

def load_data():
    goal_path = "...\Data\goalSentences.txt"
    group_path = "...\Data\groupSentences.txt"
    inter_path = "...\interactionSentences.txt"
    goal_sentences = []
    group_sentences = []
    inter_sentences = []
    with open(goal_path, 'r') as file:
        sentences = file.readlines()
    for sentence in sentences:
        goal_sentences.append(sentence.strip())  
    with open(group_path, 'r') as file:
        sentences = file.readlines()
    for sentence in sentences:
        group_sentences.append(sentence.strip()) 
    with open(inter_path, 'r') as file:
        sentences = file.readlines()
    for sentence in sentences:
        inter_sentences.append(sentence.strip()) 
    labels = torch.hstack([torch.zeros(len(goal_sentences)),torch.ones(len(group_sentences)),(torch.ones(len(inter_sentences)) * 2)])
    all_sentences = goal_sentences + group_sentences + inter_sentences
    return all_sentences, labels
def preprocess_data(text, tokenizer):
    inputs = torch.zeros(len(text),max_seq_len).long()
    for i in range(len(text)):
        input_ids = tokenizer(text[i], return_tensors="pt").input_ids

        inputs[i,0:input_ids.shape[1]] = input_ids
    return inputs


# LOAD SENTENCES
def count_max_words():
    goal_path = "...\Data\goalSentences.txt"
    group_path = "...\Data\groupSentences.txt"
    inter_path = "...\Data\interactionSentences.txt"
    goal_sentences = []
    group_sentences = []
    inter_sentences = []
    with open(goal_path, 'r') as file:
        sentences = file.readlines()
    for sentence in sentences:
        goal_sentences.append(sentence.strip('""'))  
    with open(group_path, 'r') as file:
        sentences = file.readlines()
    for sentence in sentences:
        group_sentences.append(sentence.strip('""')) 
    with open(inter_path, 'r') as file:
        sentences = file.readlines()
    for sentence in sentences:
        inter_sentences.append(sentence.strip('""')) 
    labels = np.hstack([np.zeros(len(goal_sentences)),np.ones(len(group_sentences)),(np.ones(len(inter_sentences)) * 2)])

    # PREPROCESS DATA
    all_sentences = goal_sentences + group_sentences + inter_sentences
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(all_sentences)
    sequences = tokenizer.texts_to_sequences(all_sentences)
    max_sentence = 0
    for seq in sequences:
        if len(seq) > max_sentence:
            max_sentence = len(seq)
    print(max_sentence)