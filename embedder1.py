import os
import pandas
import faiss
import numpy
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)

embedder = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    api_key=os.environ["DASHSCOPE_API_KEY"]
)


df= pandas.read_csv(r"D:\project\Qwen-proj\documents\运动鞋店铺知识库.txt", header=None)


for i in range(0, len(df)):
    result_embeddings = embedder.get_text_embedding_batch(df.loc[i,0])
    vectors = numpy.array(result_embeddings, dtype='float32')
    if i==0:
        index = faiss.IndexFlatL2(vectors.shape[1])
    else:
        index.add(vectors)

faiss.write_index(index, r"D:\project\Qwen-proj\documents\faiss1.index")
