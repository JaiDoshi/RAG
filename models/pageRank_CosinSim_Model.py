import os
from scipy.sparse import csr_matrix
import itertools
from typing import Dict, List
import nltk
from models.utils import trim_predictions_to_max_token_length
import numpy as np
import networkx as nx
import torch
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

class RAGModel:
    def __init__(self):
        """
        Initialize the RAGModel with necessary models and configurations.
        """
        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            "models/sentence-transformers/all-MiniLM-L6-v2", device="cuda"
        )

        # Define the number of context sentences to consider for generating an answer.
        self.num_context = 10
        # Set the maximum length for each context sentence in characters.
        self.max_ctx_sentence_length = 1000

        # Template for formatting the input to the language model, including placeholders for the question and references.
        self.prompt_template = """
        ### Question
        {query}

        ### References
        {references}

        ### Answer
        """

        # Configuration for model quantization to improve performance, using 4-bit precision.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )

        # Specify the large language model to be used.
        model_name = "models/meta-llama/Llama-2-7b-chat-hf"

        # Load the tokenizer for the specified model.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the large language model with the specified quantization configuration.
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )

        # Initialize a text generation pipeline with the loaded model and tokenizer.
        self.generation_pipe = pipeline(
            task="text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            max_new_tokens=10,
        )

    def select_top_sentences(self, sentences, query, num_sentences):
        """
        Select the top relevant sentences based on the query and sentence embeddings.
        """
    
        sentence_embeddings = self.sentence_model.encode(sentences, normalize_embeddings=True)
        query_embedding = self.sentence_model.encode(query, normalize_embeddings=True)

        # Compute the cosine scores
        cosine_scores = np.dot(sentence_embeddings, query_embedding)
        # Sort the sentences based on the cosine scores
        sorted_indices = np.argsort(cosine_scores)[::-1]
        sorted_sentences = [sentences[idx] for idx in sorted_indices[:num_sentences]]
        sorted_scores = cosine_scores[sorted_indices[:num_sentences]]

        # Create a sparse similarity matrix
        n = len(sentences)
        row_indices, col_indices, data = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                similarity = np.dot(sentence_embeddings[i], sentence_embeddings[j])
                row_indices.append(i)
                col_indices.append(j)
                data.append(similarity)
                row_indices.append(j)
                col_indices.append(i)
                data.append(similarity)

        #similarity_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        similarity_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

        # Compute the transition matrix
        row_sums = similarity_matrix.sum(axis=1).A.ravel()  # Convert to a 1D NumPy array
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Replace zeros with ones
        transition_matrix = (similarity_matrix + np.eye(n)) / row_sums[:, np.newaxis]

        # Compute PageRank scores
        pr_scores = np.linalg.eigh(transition_matrix.T)[1][:, -1]

        # Sort sentences based on PageRank scores
        sorted_pagerank = sorted(zip(pr_scores, sorted_sentences), reverse=True)
        top_sentences = [sentence for _, sentence in sorted_pagerank[:num_sentences]]

        return top_sentences


    def expand_query(self, query, top_sentences):
        """
        Expand the query by adding relevant keywords from the top sentences.
        """
        # Generate embeddings for the query and top sentences
        query_embedding = self.sentence_model.encode(query, normalize_embeddings=True)
        sentence_embeddings = self.sentence_model.encode(top_sentences, normalize_embeddings=True)

        # Calculate cosine similarity between query and sentence embeddings
        cosine_scores = (sentence_embeddings * query_embedding).sum(1)
        top_keywords = []

        # Select top keywords or phrases from the most relevant sentences
        for sentence, score in sorted(zip(top_sentences, cosine_scores), key=lambda x: x[1], reverse=True):
            for phrase in nltk.ngrams(sentence.split(), n=2):
                phrase_embedding = self.sentence_model.encode(" ".join(phrase), normalize_embeddings=True)
                if sum(util.pytorch_cos_sim(phrase_embedding, query_embedding) > 0.5) > 0:
                    top_keywords.append(" ".join(phrase))

        # Expand the original query with the top keywords
        expanded_query = query + " " + " ".join(top_keywords[:3])
        return expanded_query


    def generate_answer(self, query: str, search_results: List[Dict]) -> str:
        """
        Generate an answer based on the provided query and a list of pre-cached search results.
        """
        # Initialize a list to hold all extracted sentences from the search results.
        all_sentences = []

        # Process each HTML text from the search results to extract text content.
        for html_text in search_results:
            # Parse the HTML content to extract text.
            soup = BeautifulSoup(
                html_text["page_result"], features="html.parser"
            )
            text = soup.get_text().replace("\n", "")
            if len(text) > 0:
                # Convert the text into sentences and extract their offsets.
                offsets = text_to_sentences_and_offsets(text)[1]
                for ofs in offsets:
                    # Extract each sentence based on its offset and limit its length.
                    sentence = text[ofs[0] : ofs[1]]
                    all_sentences.append(
                        sentence[: self.max_ctx_sentence_length]
                    )
            else:
                # If no text is extracted, add an empty string as a placeholder.
                all_sentences.append("")

        # Select the top relevant sentences using PageRank
        top_sentences = self.select_top_sentences(all_sentences, query, self.num_context)

        # Expand the query with relevant keywords from the top sentences
        expanded_query = self.expand_query(query, top_sentences)

        # Format the top sentences as references in the model's prompt template.
        references = "\n".join(f"<DOC>\n{sentence}\n</DOC>" for sentence in top_sentences)
        final_prompt = self.prompt_template.format(
            query=expanded_query, references=references
        )

        # Generate an answer using the formatted prompt.
        result = self.generation_pipe(final_prompt)
        result = result[0]["generated_text"]

        try:
            # Extract the answer from the generated text.
            answer = result.split("### Answer\n")[-1]
        except IndexError:
            # If the model fails to generate an answer, return a default response.
            answer = "I don't know"

        # Trim the prediction to a maximum of 75 tokens (this function needs to be defined).
        trimmed_answer = trim_predictions_to_max_token_length(answer)

        return trimmed_answer