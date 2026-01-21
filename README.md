# RecencyQA
UIBK | Current Topic of Computer Science | RecencyQA

## Future Work
As mentioned in the conclusion part of the paper, it would be interesting to compare human labeling vs LLM labeling on our (or similar) dataset. The problem is to find a way to get human labels in a cost-effective manner. Therefore the idea and first layout of project was created, you can check it out at [recency-labeling-page](https://recency-labeling-page.vercel.app). (*GitHub not yet public.*)
The labeling for each question + context combination is stored in a MongoDB database. In this way, afterwards the labels can be exported as JSONL files similar to the model outputs and then be compared directly.
