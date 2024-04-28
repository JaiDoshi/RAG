#from models.dummy_model import DummyModel
#UserModel = DummyModel

# Uncomment the lines below to use the Vanilla LLAMA baseline
#from models.vanilla_llama_baseline import ChatModel 
#UserModel = ChatModel


# Uncomment the lines below to use the RAG LLAMA baseline
#from models.rag_llama_baseline_nocap import RAGModel
#UserModel = RAGModel

# Uncomment the lines below to use the RAG LLAMA baseline
#from models.rag_llama_baseline_nocap_prompt3 import RAGModel
#UserModel = RAGModel

#from models.rag_llama_nocap_faiss_bnb_prompt_sbert import RAGModel
#UserModel = RAGModel

#from models.rag_llama_chromadb_langchain import RAGModel
#UserModel = RAGModel

#from models.rag_llama_compress_retrieval import RAGModel
#UserModel = RAGModel

#from models.rag_llama_ensemble import RAGModel
#UserModel = RAGModel

from models.rag_llama_multiquery import RAGModel
UserModel = RAGModel
