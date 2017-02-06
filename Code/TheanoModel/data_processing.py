import numpy as np

def process_batch(batch_question, reverse=False, position='begin'):

    question_length = []
    for question in batch_question:
        question_length.append(len(question))
    max_length = max(question_length)

    input_idx = np.zeros((max_length, batch_question.shape[0]), dtype='int32')
    input_mask = np.zeros((max_length, batch_question.shape[0]), dtype='float32')
    
    dir = 1
    if reverse:
        dir = -1

    for i, question in enumerate(batch_question):
        if position=='begin':
            input_idx[:question_length[i], i] = question[:question_length[i]][::dir]
            input_mask[:question_length[i], i] = 1.0
        else:
            input_idx[-question_length[i]:, i] = question[:question_length[i]][::dir]
            input_mask[-question_length[i]:, i] = 1.0
    
    return input_idx, input_mask
