import json
import pickle
def save_binary_q(location):
    with open(f'CLEVR_v1.0/questions/CLEVR_{location}_questions.json','r') as file:
        data = json.load(file)
        binary_questions = [[]]
        print("json loaded")
        for question_pack in data['questions']:
            image_index = question_pack['image_index']
            question = question_pack['question']
            answer = question_pack['answer']
            if (answer != 'no' and answer != 'yes'):
                if (len('answer') == 0):
                    print('zero')
                    continue
                if (answer[0] == 'Y' or answer[0] == 'N'):
                    print(answer)
                continue
            if (len(binary_questions[0]) != 0):
                if (binary_questions[-1][0] != image_index):
                    binary_questions.append([image_index,(question,answer)])
                else:
                    binary_questions[-1].append((question,answer))
            else:
                binary_questions.append([image_index,(question,answer)])
        with open(f'binary_questions_{location}', 'wb') as f:
            pickle.dump(binary_questions, f)

def load_binary_q(location):
    with open(f'binary_questions_{location}', 'rb') as f:
        binary_questions = pickle.load(f)  # Load the pickled data
    print(binary_questions)
#save_binary_q('train')
load_binary_q('train')
        