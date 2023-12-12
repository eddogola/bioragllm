from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import Ollama
from llama_index import SimpleDirectoryReader, ServiceContext, KeywordTableIndex
from llama_index.evaluation import CorrectnessEvaluator, FaithfulnessEvaluator, RelevancyEvaluator
import pandas as pd
import os

# loads BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("HF embedding model loaded")

llm = Ollama(model="vicuna", temperature=0)

print("Ollama vicuna model loaded")

DATA_PATH = os.path.join("/Users/ogola/tinyml/project/bioragllm", "ps5/data")
docs = SimpleDirectoryReader(DATA_PATH).load_data()

print("Data loaded in `docs`")

service_cxt = ServiceContext.from_defaults(llm=llm, embed_model=embed_model,
                                           chunk_size=512)

print("Service context created")

index = KeywordTableIndex.from_documents(docs, service_context=service_cxt)

print("Index created")

query_engine = index.as_query_engine()

print("Query engine created")

QNA_PATH = os.path.join("/Users/ogola/tinyml/project/bioragllm", "ps5/qna.csv")
df = pd.read_csv(QNA_PATH)

print("questions and answers data loaded")

# correctness_evaluator = CorrectnessEvaluator(service_context=service_cxt)
faithfulness_evaluator = FaithfulnessEvaluator(service_context=service_cxt)
relevancy_evaluator = RelevancyEvaluator(service_context=service_cxt)

data = []
for i in range(5):#range(len(df)):
    question = df.iloc[i, 1]
    choices = df.iloc[i, 2]
    qna = f"Give the correct answer choice number to the following question\n{question}\n {choices}"
    response = query_engine.query(qna)

    print(f"Generated response to Question {i+1}")

    # faithfulness evaluation
    faithfulness_eval = faithfulness_evaluator.evaluate_response(query=qna, response=response)
    faithfulness_flag = faithfulness_eval.passing

    print(f"Generated faithfulness eval for Question {i+1}")

    # relevancy evaluation
    relevancy_eval = relevancy_evaluator.evaluate_response(query=qna, response=response)
    relevancy_flag = relevancy_eval.passing

    print(f"Generated relevancy eval for Question {i+1}")

    # correctness evaluation
    # correct_answer = df.iloc[i, 3]
    # correctness_eval = correctness_evaluator.evaluate(query=qna, response=response,
    #                                                   reference=correct_answer)
    # correctness_score = correctness_eval.score


    # data.append({"question": question, "choices": choices, "qna": qna, "response": response,
    #                       "correctness_score": correctness_score, "faithfulness_flag": faithfulness_flag,
    #                       "relevancy_flag": relevancy_flag}, index=[0])

    data.append({"question": question, "choices": choices, "qna": qna, "response": response,
                          "faithfulness_flag": faithfulness_flag,
                          "relevancy_flag": relevancy_flag})

    print(f"Question {i+1} done")

df = pd.DataFrame(data)
RESULTS_PATH = os.path.join("/Users/ogola/tinyml/project/bioragllm", "ps5/results-vicuna.csv")
df.to_csv(RESULTS_PATH, mode='a', header=False)