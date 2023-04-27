import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from llama_index import GPTSimpleVectorIndex, ServiceContext, SimpleDirectoryReader, LLMPredictor

load_dotenv()
documents_filepath = "documents"
index_filepath = "index_store/index.json"

# インデックスファイルが存在するかどうか
if os.path.exists(index_filepath):
    # インデックスの読み込み
    vector_index = GPTSimpleVectorIndex.load_from_disk(index_filepath)
else:
    # インデックスの作成・保存
    documents = SimpleDirectoryReader(documents_filepath).load_data()
    service_context = ServiceContext.from_defaults(llm_predictor = LLMPredictor(llm= ChatOpenAI( model_name="gpt-3.5-turbo")))
    vector_index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    vector_index.save_to_disk(index_filepath)

# 質問応答
query = "ゼルダの伝説 ティアーズオブザキングダムの発売日の日付だけを教えてください。回答は日本語でお願いします。"
answer = vector_index.query(query)
print(answer)
