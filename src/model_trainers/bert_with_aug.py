from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_ID = "bert-base-uncased"  # BERT encoder w/ QA head via AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)
