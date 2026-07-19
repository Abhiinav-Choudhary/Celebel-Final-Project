import tensorflow as tf
import project_config as config
from siamese_network import build_embedding_network, extract_siamese_embeddings

print("Rebuilding model and loading weights...")
embedding_network = build_embedding_network()
embedding_network.load_weights(config.SIAMESE_MODEL_PATH)

print("Extracting embeddings...")
extract_siamese_embeddings(embedding_network)
print("Done!")
