import requests
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter


class Indexing:
    def __init__(self, url, client):
        self.url = url
        self.client = client
        self.file_paths = self._get_repo_files_names()
        self.dataset = self._generate_dataset()


    def _strip_github_url(self):
        parts = self.url.rstrip("/").split("/")

        if len(parts) < 2:
            print("Invalid GitHub repository URL.")
            return
        owner, repo = parts[-2], parts[-1]
        return owner, repo

    def _get_repo_files_names(self):
        owner, repo = self._strip_github_url()

        api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
        headers = {"Accept": "application/vnd.github.v3+json"}

        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if "tree" in data:
                file_paths = [item["path"] for item in data["tree"] if item["type"] == "blob"]
                return file_paths
        else:
            print(f"Error: Unable to fetch repository data (Status Code: {response.status_code}")
            return

    def _get_file_content(self, file_path):
        owner, repo = self._strip_github_url()

        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}"
        response = requests.get(raw_url)

        if response.status_code == 200:
            return response.text
        else:
            print(f"Error: Unable to fetch file content (Status Code: {response.status_code}")
            return None

    def _generate_code_summary(self, code, model="gpt-4o-mini"):
        prompt = """
            You are a knowledgeable about a coding. 
            Your users are giving you inside of the files of github repository. It could be code, or text. Your goal is to summarize the functionality of the code/text. 
            It should not be too long, but should not ommit any part of the functionality. When you summarize, think about questions that the user might have about the repository,
            and how you summary can help to identify that this specific file, might be helpful to answer user question. 
                        """

        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user", "content": code}
        ]
        response = self.client.chat.completions.create(
            model=model,
            messages=messages
        )
        content = response.choices[0].message.content
        content = content.split("\n")
        return content

    def _generate_dataset(self):
        dataset = pd.DataFrame(columns=['file_path', 'summary'])
        for i ,f in enumerate(self.file_paths):
                # code = self._get_file_content(f)
                # summary = self._generate_code_summary(code)
                # splitted_summary = self._split_text(summary)
                dataset.loc[len(dataset)] = [f, f]
                print("dataset entry generated")
        return dataset

    def _split_text(self, text):
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""], chunk_size=400, chunk_overlap=0
        )
        character_split_texts = character_splitter.split_text("\n\n".join(text))

        # chunks into tokens size
        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=0, tokens_per_chunk=256,
        )
        token_split_texts = []
        for text in character_split_texts:
            token_split_texts += token_splitter.split_text(text)
        return token_split_texts


