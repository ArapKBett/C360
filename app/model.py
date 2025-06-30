import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"  # Replace with URL-specific model if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_url(url):
    inputs = tokenizer(url, return_tensors="np", padding=True, truncation=True, max_length=128)
    input_ids = jnp.array(inputs['input_ids'])
    attention_mask = jnp.array(inputs['attention_mask'])
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    probabilities = jax.nn.softmax(logits, axis=-1)
    prediction = jnp.argmax(probabilities, axis=-1).item()
    confidence = probabilities[0, prediction].item()
    return prediction, confidence
