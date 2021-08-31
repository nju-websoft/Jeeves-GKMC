## GKMC

The structure of `GKMC.json` is as follows:
```json
[
    {
        "id": "The id of question",
        "scenario": "The text of scenario",
        "question": "The text of question",
        "A": "The text of option A",
        "B": "The text of option B",
        "C": "The text of option C",
        "D": "The text of option D",
        "answer": "The correct answer of this question",
        "paragraph_a": [
            "A list of relevant paragraphs for option A annotated by human."
            {
                "p_id": "The id of paragraph",
                "content": "The text of paragraph"
            },
            ...
        ],
        "paragraph_b":[...],
        "paragraph_c":[...],
        "paragraph_d":[...]
    },
    {
        ...
    }
]
```


## The Usage of JEEVES

1. Save the pre-trained language model in the `/pytorch_jeeves/data` folder, such as ERNIE and BERT-wwm-ext.
2. Save the training data with its candidate paragraphs in `/pytorch_jeeves/data/GeoSQA`, `/pytorch_jeeves/data/GKMC` and `/pytorch_jeeves/data/GH577`
3. To train the JEEVES model, we can use the following command:
``` python
python run_jeeves.py
```
4. To predict the answer, we can use Lucene to efficiently compute retrieval score with word weights. We use the version of Lucene-6.2.0. .

### Python packages
- Pytorch
- jieba 
