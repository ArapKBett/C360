import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"  # Replace with URL-specific model if available
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
except Exception as e:
    raise Exception(f"Failed to load model: {str(e)}")

def predict_url(url):
    try:
        # Tokenize URL
        inputs = tokenizer(url, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Convert to JAX arrays
        input_ids = jnp.array(inputs['input_ids'].numpy())
        attention_mask = jnp.array(inputs['attention_mask'].numpy())
        
        # Run inference with PyTorch model
        with torch.no_grad():
            logits = model(input_ids=torch.tensor(np.array(input_ids)), 
                          attention_mask=torch.tensor(np.array(attention_mask))).logits
        
        # Convert logits to JAX for processing
        logits = jnp.array(logits.numpy())
        probabilities = jax.nn.softmax(logits, axis=-1)
        
        # Get prediction and confidence
        prediction = jnp.argmax(probabilities, axis=-1).item()
        confidence = probabilities[0, prediction].item()
        
        return prediction, confidence
    except Exception as e:
        raise Exception(f"Inference failed: {str(e)}")