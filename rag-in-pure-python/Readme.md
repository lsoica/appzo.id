---
layout: sidebar-layout
title:  "RAG in pure Python"
date:   2024-08-16 11:08:03 +0200
---
# RAG

## Chunking and embedding

A document is chunked first. The strategy could be as simple as splitting on paragraphs (e.g. \n\n).

Then for each chunk a single vector embedding is returned by the model. Because the model is built to compute one embedding for each token, a strategy is required to combine the embeddings of the tokens in each chunk into a single embedding. The simplest approach would be to take the average of all the vectors.

A pooling mechanism is applied to reduce the token-level embeddings into a single vector. This is often done using:
 - Mean pooling: Averaging the embeddings of all tokens.
 - CLS token pooling: Using the special [CLS] token in models like BERT as the aggregate representation of the entire input sequence.

## Querying

 documents -> chunks -> embedding vector
                                              dot product -> most similar n chunks -> feed the model with the query and the top n paragraphs -> return the results.
 query               -> embedding vector

 An embedding is computed for the query string as well, then the query vector is compared to each chunk vector. The closest match(s) are returned and now we know the top n documents that are most similar to the query. We can feed these documents into the model to get the results.

## Reference

 [RAG in pure python demo](https://www.youtube.com/watch?v=bmduzd1oY7U&ab_channel=PromptEngineering)

## Implementation

{% jupyter_notebook "rag.ipynb" %}