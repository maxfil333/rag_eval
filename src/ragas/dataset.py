from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

texts = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")

# print(texts)
# print(texts['passages']['passage'][0])
# print(texts['passages']['passage'][1])
#
# print(qa)
# print(qa['test']['question'][0])
# print(qa['test']['answer'][0])


docs = [texts['passages']['passage'][i] for i in range(5)]
print(docs)
