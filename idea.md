

Here is the prompt in METIS:
```
For the given query = {get.query()}: Analyse  the language and internal structure of the query and provide the following information : 4  5 1. Does it needs joint reasoning across  multiple documents or not. 6 2. Provide a complexity profile for the  query: 7 Complexity: High/Low \n \ 8 Joint Reasoning needed: Yes/No \n " 9 3. Does this query need input chunks to  be summarized and if yes , provide a range in words for the summarized chunks. 10 4. How many pieces of information is  needed to answer the query? 11  12 database_metadata = { get . metadata () } 13 chunk_size = { get . chunk_size () } 14 15 Estimate the query profile along with the  database_metadata and chunk_size to provide the output.
```

The refined idea for LLM optimizer is:
First, the built tree should be stored in pgsql, and each node should have a guanularity columnn (the depth of the node).

There should be a metadata telling each guanularity's meanings.

When the optimizer receives the query, it would check whether it should choose:
- entire collapsed tree
- partial collapsed tree based on guanularity check
- tree treversal,
- ...