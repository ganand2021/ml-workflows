import random
import string

def generate_model_signature(length):
  """Generates a random hash string of the given length."""
  characters = string.ascii_letters + string.digits
  return ''.join(random.choice(characters) for i in range(length))