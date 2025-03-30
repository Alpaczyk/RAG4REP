import os

from dotenv import load_dotenv

from Indexing import Indexing
from sentence_transformers import SentenceTransformer
import chromadb
from Evaluator import Evaluator
import numpy as np
from openai import OpenAI


class RAGPipeline:
    def __init__(self, url, no_retrieved_file_names=10):
        load_dotenv()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection_name = "files"
        self.url = url
        self.no_retrieved_file_names = no_retrieved_file_names
        self.dataset = Indexing(self.url, self.openai_client).dataset
        self.embedding_function = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client()
        self.evaluator = Evaluator()

        self.initialize_pipeline()



    def initialize_pipeline(self):
        """Ensures the collection is created only once and filled with embeddings."""
        try:
            # Try to get the existing collection
            self.collection = self.chroma_client.get_collection(self.collection_name)
            print("Collection already exists. Skipping creation and data insertion.")

        except Exception:  # Handle case where collection doesn't exist
            print("Collection does not exist. Creating a new one...")
            self._create_collection()  # Create the collection
            self._dataset_to_collection()  # Populate with embeddings
            print("Collection populated with embeddings.")

        self.evaluate()  # Continue with evaluation

    def _create_collection(self):

        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
        )
        print("collection created")

    def _get_embeddings(self, text):
        return self.embedding_function.encode(text)

    def _dataset_to_collection(self):
        for i in range(0, len(self.dataset)):
            file_names = self.dataset['file_path'][i]
            documents = self.dataset['summary'][i]
            if not isinstance(documents, list):
                documents = [documents]

            embeddings = self._get_embeddings(documents)

            for j, (chunk, embedding) in enumerate(zip(documents, embeddings)):
                id = f"{i}_{j}"

                self.collection.add(
                    documents=chunk,
                    ids=id,
                    metadatas={"file_name": file_names, "chunk": j},
                    embeddings=embedding
                )
                print("Embeddings added to collection")

    def _generate_multi_query(self, query, model="gpt-4o-mini"):
        prompt = """
            You are a knowledgeable about a github repository. 
            Your users are inquiring about repository. 
            For the given question, propose up to three related questions to assist them in finding the files from github repository related to the question. 
            Provide concise, single-topic questions (withouth compounding sentences) that cover various aspects of the topic. 
            Ensure each question is complete and directly related to the original inquiry. 
            List each question on a separate line without numbering.
                        """

        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user", "content": query}
        ]
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        content = response.choices[0].message.content
        content = content.split("\n")
        return content


    def _create_query_embedding(self, query):
        multi_query = self._generate_multi_query(query)
        return self._get_embeddings(multi_query)

    def _retrieve_file_names(self, query):
        results = self.collection.query(
            query_embeddings=self._create_query_embedding(query),
            n_results=self.no_retrieved_file_names
        )
        file_names = [entry.get("file_name") for meta in results["metadatas"] for entry in meta]
        return file_names

    def query_input(self, query):
        return self._retrieve_file_names(query)

    def evaluate(self):
        recall_scores = []
        for i in range(len(self.evaluator.queries_to_evaluate)):
            retrieved_file_names = self._retrieve_file_names(self.evaluator.queries_to_evaluate['question'][i])
            relevant_file_names = self.evaluator.queries_to_evaluate['files'][i]
            recall_scores.append(self.evaluator.evaluate_single_question(retrieved_file_names, relevant_file_names))
        print(np.mean(recall_scores))

