import os
from typing import Dict, List

import numpy as np
import torch
from blingfire import text_to_sentences_and_offsets
from haystack.document_stores import FAISSDocumentStore, ElasticsearchDocumentStore
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever, PromptNode
from haystack.nodes import EmbeddingRetriever, TfidfRetriever
from haystack import Document
from haystack.pipelines import Pipeline
from haystack.nodes import FARMReader, TransformersReader, TableReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
#from haystack.utils import launch_milvus
#from haystack.document_stores import MilvusDocumentStore
from bs4 import BeautifulSoup
from models.utils import trim_predictions_to_max_token_length
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)



######################################################################################################
######################################################################################################
###
### IMPORTANT !!!
### Before submitting, please follow the instructions in the docs below to download and check in :
### the model weighs.
###
###  https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/download_baseline_model_weights.md
###
###
### DISCLAIMER: This baseline has NOT been tuned for performance
###             or efficiency, and is provided as is for demonstration.
######################################################################################################


# Load the environment variable that specifies the URL of the MockAPI. This URL is essential
# for accessing the correct API endpoint in Task 2 and Task 3. The value of this environment variable
# may vary across different evaluation settings, emphasizing the importance of dynamically obtaining
# the API URL to ensure accurate endpoint communication.

# Please refer to https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/crag-mock-api
# for more information on the MockAPI.
#
# **Note**: This environment variable will not be available for Task 1 evaluations.
CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")


class RAGModel:
    def __init__(self):
        """
        Initialize the RAGModel with necessary models and configurations.

        This constructor sets up the environment by loading sentence transformers for embedding generation,
        a large language model for generating responses, and tokenizer for text processing. It also initializes
        model parameters and templates for generating answers.
        """
        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        # self.sentence_model = SentenceTransformer(
        #     "sentence-transformers/all-MiniLM-L6-v2", device="cuda"
        # )

        # # Define the number of context sentences to consider for generating an answer.
        # self.num_context = 10
        # # Set the maximum length for each context sentence in characters.
        self.max_ctx_sentence_length = 1000

        # # Template for formatting the input to the language model, including placeholders for the question and references.
        # self.prompt_template = """
        # ### Question
        # {query}

        # ### References 
        # {references}

        # ### Answer
        # """

        # # Configuration for model quantization to improve performance, using 4-bit precision.
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=False,
        # )

        # # Specify the large language model to be used.
        # model_name = "meta-llama/Llama-2-7b-chat-hf"

        # # Load the tokenizer for the specified model.
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # # Load the large language model with the specified quantization configuration.
        # self.llm = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     device_map="auto",
        #    # quantization_config=bnb_config,
        #     torch_dtype=torch.float16,
        # )

        # # Initialize a text generation pipeline with the loaded model and tokenizer.
        # self.generation_pipe = pipeline(
        #     task="text-generation",
        #     model=self.llm,
        #     tokenizer=self.tokenizer,
        #     max_new_tokens=75,
        # )

    def generate_answer(self, query: str, search_results: List[Dict]) -> str:
        """
        Generate an answer based on the provided query and a list of pre-cached search results.

        Parameters:
        - query (str): The user's question.
        - search_results (List[Dict]): A list containing the search result objects,
        as described here:
          https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail

        Returns:
        - str: A text response that answers the query. Limited to 75 tokens.

        This method processes the search results to extract relevant sentences, generates embeddings for them,
        and selects the top context sentences based on cosine similarity to the query embedding. It then formats
        this information into a prompt for the language model, which generates an answer that is then trimmed to
        meet the token limit.
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


        # if os.path.exists('faiss_document_store.db'):
        #     document_store = FAISSDocumentStore.load('faiss_document_store.db')
        # else:
        document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True, sql_url="sqlite://", vector_dim=384)
        #launch_milvus()
        #document_store = MilvusDocumentStore()
    #     dicts = [
    # {
    #     'content': "Voldermort died in the hands of Snape",
    #     'meta': {'name': 'doc1'}
    # },
    #  {
    #     'content': "Severus Snape killed Tom Riddle aka Lord Voldermort",
    #     'meta': {'name': 'doc2'}
    # },
    #     ]
        dicts = []
        cnt = 0
        for sentence in all_sentences:
            cnt +=1
            dicts.append(
                {
                    'content': sentence,
                    'meta': {'name': cnt}
                }
            )

        # external_doc = [Document(content="Voldermort died in the hands of Snape")
        #                 , Document(content = "Severus Snape killed Tom Riddle aka Lord Voldermort")]


        # Initialize DPR Retriever to encode documents, encode question and query documents
    #     retriever = DensePassageRetriever(
    #     document_store=document_store,
    #     query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    #     passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    #     max_seq_len_query=64,
    #     max_seq_len_passage=256,
    #     batch_size=16,
    #     use_gpu=True,
    #     embed_title=True,
    #     use_fast_tokenizers=True
    # )
        retriever = TfidfRetriever(
    document_store=document_store
    )

        document_store.delete_documents()
        document_store.write_documents(dicts)   

        #document_store.update_embeddings(retriever=retriever)

        reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
        pipe = ExtractiveQAPipeline(reader, retriever)
        prediction = pipe.run(
    query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        )   
        #print(prediction)

        print_answers(prediction, details="minimum")

        return


        prompt_node = PromptNode()

        pipe = Pipeline()
        pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
        pipe.add_node(component=prompt_node, name="prompt_node", inputs=["Retriever"])

        output = pipe.run(query=query)

        print(output)

        return

        # Generate embeddings for all sentences and the query.
        all_embeddings = self.sentence_model.encode(
            all_sentences, normalize_embeddings=True
        )
        query_embedding = self.sentence_model.encode(
            query, normalize_embeddings=True
        )[None, :]

        # Calculate cosine similarity between query and sentence embeddings, and select the top sentences.
        cosine_scores = (all_embeddings * query_embedding).sum(1)
        top_sentences = np.array(all_sentences)[
            (-cosine_scores).argsort()[: self.num_context]
        ]

        # Format the top sentences as references in the model's prompt template.
        references = ""
        for snippet in top_sentences:
            references += "<DOC>\n" + snippet + "\n</DOC>\n"
        references = " ".join(
            references.split()[:500]
        )  # Limit the length of references to fit the model's input size.
        final_prompt = self.prompt_template.format(
            query=query, references=references
        )

        print(final_prompt)

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
        return answer
        
        trimmed_answer = trim_predictions_to_max_token_length(answer)

        return trimmed_answer

