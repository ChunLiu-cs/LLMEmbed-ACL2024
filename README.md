# LLMEmbed: Rethinking Lightweight LLM's Genuine Function in Text Classification
This code is for the LLMEmbed paper accepted in the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)
*DOI: *

## llama2_embedding / bert_embedding / roberta_embedding
The `rep_extract.py` uses language model to extract the representation of `dataset` and saves the representation as `.pt` file.

## MyDataset
`MyDataset.py` reads the representation from `.pt` file.

## DownstreamModel
`DownstreamModel.py` is for the **co-occurence pooling**.

## 📜Citation

This work has been accepted to [ACL-2024](DOI: ), please cite the paper if you use LLMEmbed or this repository in your research.
Thank you very much 😉


