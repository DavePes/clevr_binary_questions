import json
import pickle

class BinaryQuestionHandler:
    @staticmethod
    def save_binary_q(location):
        """
        Saves binary questions (questions with 'yes' or 'no' answers) from a JSON file to a binary file.
        
        Args:
            location (str): The location identifier to locate the JSON file and save the binary file.
        """
        try:
            with open(f'CLEVR_v1.0/questions/CLEVR_{location}_questions.json', 'r') as file:
                data = json.load(file)
                binary_questions = []
                print("JSON loaded successfully.")

                for question_pack in data['questions']:
                    image_index = question_pack['image_index']
                    question = question_pack['question']
                    answer = question_pack['answer']

                    if answer not in {'yes', 'no'}:
                        continue

                    if not binary_questions or binary_questions[-1][0] != image_index:
                        binary_questions.append([image_index, (question, answer)])
                    else:
                        binary_questions[-1].append((question, answer))

                with open(f'binary_questions_{location}', 'wb') as f:
                    pickle.dump(binary_questions, f)
                    print(f"Binary questions saved to binary_questions_{location}")

        except Exception as e:
            print(f"An error occurred while saving binary questions: {e}")

    @staticmethod
    def load_binary_q(location):
        """
        Loads binary questions from a binary file and prints them.
        
        Args:
            location (str): The location identifier to locate the binary file.
        """
        try:
            with open(f'binary_questions_{location}', 'rb') as f:
                binary_questions = pickle.load(f)
                print("Binary questions loaded successfully:")
                #print(binary_questions)
                return binary_questions
        except FileNotFoundError:
            print(f"The file binary_questions_{location} does not exist.")
        except Exception as e:
            print(f"An error occurred while loading binary questions: {e}")

