import pandas as pd
import os

class Evaluator:
    def __init__(self,  evaluation_data_path="./Data/escrcpy-commits-generated.json"):
        self.evaluation_data_path = evaluation_data_path

        self._get_files_and_queries()

    def _get_files_and_queries(self):
        if os.path.exists(self.evaluation_data_path):
            self.queries_to_evaluate = pd.read_json(self.evaluation_data_path)
        else:
            self.queries_to_evaluate = pd.DataFrame(columns=['files', 'question'])



    def evaluate_single_question(self, retrieved_file_names, relevant_file_names):
        print(f"Retrieved files: {retrieved_file_names}")
        print(f"relevatant files: {relevant_file_names}")
        nominator_recall = len([value for value in relevant_file_names if value in retrieved_file_names])
        denominator_recall = len(relevant_file_names)
        return nominator_recall / denominator_recall

