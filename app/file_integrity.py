import hashlib
import os

def check_file_integrity(file_path):
    if not os.path.exists(file_path):
        raise Exception("File not found")
    
    # Calculate SHA-256 hash
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    current_hash = sha256_hash.hexdigest()
    
    # Placeholder: Compare with stored hash (in practice, store hashes securely)
    stored_hash = "stored_hash_placeholder"  # Replace with actual hash storage
    return {
        'file': file_path,
        'current_hash': current_hash,
        'is_unchanged': current_hash == stored_hash
    }
