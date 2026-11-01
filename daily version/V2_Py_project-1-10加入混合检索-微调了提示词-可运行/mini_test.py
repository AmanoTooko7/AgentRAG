import bm25s

corpus = ["hello world", "hello python"]
retriever = bm25s.BM25()
retriever.index(bm25s.tokenize(corpus))

# 🔥 关键修改：显式指定 k=1
results = retriever.retrieve(bm25s.tokenize(["hello"]), k=1)

print("Type:", type(results))
print("Attributes:", dir(results))
print("Content:", results)