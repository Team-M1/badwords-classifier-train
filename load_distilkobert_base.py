from transformers import DistilBertForSequenceClassification
from tokenization_kobert import KoBertTokenizer

model = DistilBertForSequenceClassification.from_pretrained("DistilBERT/distilkobert")
tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert", model_max_length=512)