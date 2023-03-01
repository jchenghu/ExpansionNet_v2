import re


def compute_num_pads(list_bboxes):
    max_len = -1
    for bboxes in list_bboxes:
        num_bboxes = len(bboxes)
        if num_bboxes > max_len:
            max_len = num_bboxes
    num_pad_vector = []
    for bboxes in list_bboxes:
        num_pad_vector.append(max_len - len(bboxes))
    return num_pad_vector


def remove_punctuations(sentences):
    punctuations = ["''", "'", "``", "`", ".", "?", "!", ",", ":", "-", "--", "...", ";"]
    res_sentences_list = []
    for i in range(len(sentences)):
        res_sentence = []
        for word in sentences[i].split(' '):
            if word not in punctuations:
                res_sentence.append(word)
        res_sentences_list.append(' '.join(res_sentence))
    return res_sentences_list


def lowercase_and_clean_trailing_spaces(sentences):
    return [(sentences[i].lower()).rstrip() for i in range(len(sentences))]


def add_space_between_non_alphanumeric_symbols(sentences):
    return [re.sub(r'([^\w0-9])', r" \1 ", sentences[i]) for i in range(len(sentences))]


def tokenize(list_sentences):
    res_sentences_list = []
    for i in range(len(list_sentences)):
        sentence = list_sentences[i].split(' ')
        while '' in sentence:
            sentence.remove('')
        res_sentences_list.append(sentence)
    return res_sentences_list


def convert_vector_word2idx(sentence, word2idx_dict):
    return [word2idx_dict[word] for word in sentence]


def convert_allsentences_word2idx(sentences, word2idx_dict):
    return [convert_vector_word2idx(sentences[i], word2idx_dict) for i in range(len(sentences))]


def convert_vector_idx2word(sentence, idx2word_list):
    return [idx2word_list[idx] for idx in sentence]


def convert_allsentences_idx2word(sentences, idx2word_list):
    return [convert_vector_idx2word(sentences[i], idx2word_list) for i in range(len(sentences))]


def tokens2description(tokens, idx2word_list, sos_idx, eos_idx):
    desc = []
    for tok in tokens:
        if tok == sos_idx:
            continue
        if tok == eos_idx:
            break
        desc.append(tok)
    desc = convert_vector_idx2word(desc, idx2word_list)
    desc[-1] = desc[-1] + '.'
    pred = ' '.join(desc).capitalize()
    return pred
