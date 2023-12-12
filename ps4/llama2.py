from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import Ollama
from llama_index import SimpleDirectoryReader, ServiceContext, KeywordTableIndex
from llama_index.evaluation import CorrectnessEvaluator, FaithfulnessEvaluator, RelevancyEvaluator
import pandas as pd
import os

# loads BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

llm = Ollama(model="llama2", temperature=0)
DATA_PATH = os.path.join("/Users/ogola/tinyml/project/bioragllm", "ps4/data")
docs = SimpleDirectoryReader(DATA_PATH).load_data()
service_cxt = ServiceContext.from_defaults(llm=llm, embed_model=embed_model,
                                           chunk_size=512)
index = KeywordTableIndex.from_documents(docs, service_context=service_cxt)
query_engine = index.as_query_engine()
QNA_PATH = os.path.join("/Users/ogola/tinyml/project/bioragllm", "ps4/qna.csv")
df = pd.read_csv(QNA_PATH)

correctness_evaluator = CorrectnessEvaluator(service_context=service_cxt)
faithfulness_evaluator = FaithfulnessEvaluator(service_context=service_cxt)
relevancy_evaluator = RelevancyEvaluator(service_context=service_cxt)

data = []
for i in range(len(df)):
    question = df.iloc[i, 1]
    choices = df.iloc[i, 2]
    qna = f"Give the correct answer choice number to the following question\n{question}\n {choices}"
    response = query_engine.query(qna)

    # correctness evaluation
    correct_answer = df.iloc[i, 3]
    correctness_eval = correctness_evaluator.evaluate(query=qna, response=response,
                                                      reference=correct_answer)
    correctness_score = correctness_eval.score
    # faithfulness evaluation
    faithfulness_eval = faithfulness_evaluator.evaluate_response(query=qna, response=response)
    faithfulness_flag = faithfulness_eval.passing

    # relevancy evaluation
    relevancy_eval = relevancy_evaluator.evaluate_response(query=qna, response=response)
    relevancy_flag = relevancy_eval.passing

    data.append({"question": question, "choices": choices, "qna": qna, "response": response,
                          "correctness_score": correctness_score, "faithfulness_flag": faithfulness_flag,
                          "relevancy_flag": relevancy_flag}, index=[0])
df = pd.DataFrame(data)
RESULTS_PATH = os.path.join("/Users/ogola/tinyml/project/bioragllm", "ps4/results.csv")
df.to_csv(RESULTS_PATH, mode='a', header=False)