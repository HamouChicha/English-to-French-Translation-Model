import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to lowercase the text
def lower_text(text):
    text = text.replace('\u202f', '')  # Remove non-breaking space
    text = text.lower()
    return text



# Function to count the number of words in a sentence
def word_count(line):
  return len(line.split())



# Function to compute the max length of the sentences
def max_sentence_length(sentences):
  return max(len(sentence.split()) for sentence in sentences)



# Function to encode and pad the sentences
def encode_sequences(tokenizer, sentences, max_sent_len):
  text_to_seq = tokenizer.texts_to_sequences(sentences) # encode sequences with integers
  text_pad_seq = pad_sequences(text_to_seq, maxlen=max_sent_len, padding='post') # pad sequences with 0
  return text_pad_seq



# function to preprocess the input sentence
def preprocess_input(sentence, tokenizer, max_length):
    sentence = lower_text(sentence)
    padded = encode_sequences(tokenizer, [sentence], max_length)
    return padded

# function to convert the predicted word indices into sentence:
def decode_output(predicted_indices, tokenizer):
    predicted_words = tokenizer.sequences_to_texts(predicted_indices)
    return predicted_words

# The translate function:
def translate(input_sentence, eng_tokenizer, fr_tokenizer, attention_model, max_eng_sent_len, max_fr_sent_len):
      # Preprocess the input
      encoder_input = preprocess_input(input_sentence, eng_tokenizer, max_eng_sent_len)

      # Prepare decoder input (start with a special start token)
      decoder_input = np.zeros((1, max_fr_sent_len))  # Shape: (1, max_length)
      decoder_input[0, 0] = fr_tokenizer.word_index['<start>']

      # Make predictions
      predicted_sentence = []
      for t in range(1, max_fr_sent_len):
          output = attention_model.predict([encoder_input, decoder_input], verbose=0)
          predicted_word_index = np.argmax(output[0, t-1, :])
          if predicted_word_index == fr_tokenizer.word_index['<end>']:
              break
          predicted_sentence.append(predicted_word_index)

          decoder_input[0, t] = predicted_word_index

      # Decode the output
      translated_sentence = decode_output([predicted_sentence], fr_tokenizer)

      return translated_sentence[0]