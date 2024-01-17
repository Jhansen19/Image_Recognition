# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:03:48 2023

@author: Jon
"""
#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: (Jonathan Hansen)
# (based on skeleton code by D. Crandall, Nov 2023)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        char_img = []
        for y in range(0, CHARACTER_HEIGHT):
            row = [1 if px[x, y] < 1 else 0 for x in range(x_beg, x_beg + CHARACTER_WIDTH)]
            char_img.append(row)
        result.append(np.array(char_img))
    return result

def load_training_letters(fname):
    # TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }


####### IMPLEMENTING SIMPLE BAYES NET #########

def parse_training_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    clean_text = []
    for line in lines:
        elements = line.split()
        words = [elements[i] for i in range(0, len(elements), 2) if elements[i] not in ["``", "''", ".", ",", ":", ";"]]
        clean_text.extend(words)

    return ' '.join(clean_text)

def calculate_priors(train_txt_fname):
    priors = {}
    total_chars = 0

    # Process the training text
    clean_training_text = parse_training_data(train_txt_fname)

    # Count each character in the cleaned text
    for char in clean_training_text:
        if char in TRAIN_LETTERS:  # Use only valid characters
            priors[char] = priors.get(char, 0) + 1
            total_chars += 1

    # Convert counts to probabilities
    for char in priors:
        priors[char] /= total_chars

    # Set the prior for space to zero
    priors[' '] = 0

    return priors

def estimate_noise_level(test_char, average_images):
    min_noise_level = float('inf')
    for char, average_image in average_images.items():
        difference = np.abs(test_char - average_image)
        noise_level = np.mean(difference)
        if noise_level < min_noise_level:
            min_noise_level = noise_level
    return min_noise_level * 100


def calculate_average_images(train_letters):
    average_images = {}
    for char, images in train_letters.items():
        average_images[char] = np.mean(images, axis=0)
    return average_images


def apply_mean_filter(image, window_size=3):
    pad_size = window_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)

    filtered_image = np.zeros_like(image)

    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            window = padded_image[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            filtered_image[i-pad_size, j-pad_size] = np.mean(window)

    return filtered_image

# Iterate through each pixel in the test and reference (training) character.
# Compare the pixels; if they match, multiply the probability by (1 - noise_level), and if they don't match, multiply by noise_level.
# Calculate the total likelihood based on the product of probabilities for all pixels.
def calculate_pixel_distance_from_center(x, y, center_x, center_y):
    # Calculate Euclidean distance from the center
    return ((x - center_x) ** 2 + (y - center_y) ** 2) ** .7

def calculate_likelihoods(test_char, train_char, noise_level, char_key, space_noise_increase_factor=1.5):
    # Check if current character is a space
    is_space = (char_key == ' ')

    # Adjust noise level for space character
    adjusted_noise_level = noise_level * space_noise_increase_factor if is_space else noise_level
    noise_factor = adjusted_noise_level / 100  # Convert percentage to a decimal
    match_factor = 1 - noise_factor

    center_x, center_y = CHARACTER_WIDTH // 2, CHARACTER_HEIGHT // 2
    max_distance = calculate_pixel_distance_from_center(0, 0, center_x, center_y)

    likelihood = 1

    for y in range(CHARACTER_HEIGHT):
        for x in range(CHARACTER_WIDTH):
            distance = calculate_pixel_distance_from_center(x, y, center_x, center_y)
            weight = 1 - (distance / max_distance)  # Weight decreases with distance

            if test_char[y, x] == train_char[y, x]:
                likelihood *= (match_factor ** weight)
            else:
                likelihood *= (noise_factor ** weight)

    return likelihood


# In two loops iterating through train_letters.items(). 
# The first loop calculates the likelihoods but doesn't do anything with them, and then 
#  you repeat the same loop again. The first loop should be removed, and the calculations should be done in the second loop.

def recognize_character(test_char, train_letters, priors, noise_level, space_noise_increase_factor=0.5):
    best_char = None
    max_posterior = 0

    for char_key, train_char in train_letters.items():
        # Adjust noise level for space character
        adjusted_noise_level = noise_level
        if char_key == ' ':
            adjusted_noise_level *= space_noise_increase_factor

        char_likelihood = calculate_likelihoods(test_char, train_char, noise_level, char_key, space_noise_increase_factor)

        # Calculate posterior for being the current character
        char_posterior = char_likelihood * priors.get(char_key, 0)

        # Compare and choose the best option
        if char_posterior > max_posterior:
            max_posterior = char_posterior
            best_char = char_key

    return best_char if best_char is not None else ' '  # or another default character


def recognize_text(test_letters, train_letters, priors, average_images):
    recognized_text = ""
    for test_char in test_letters:
        # Apply mean filter to smooth the test character
        smoothed_test_char = apply_mean_filter(test_char)

        # Dynamic noise level estimation for each smoothed test character
        dynamic_noise_level = estimate_noise_level(smoothed_test_char, average_images)

        # Recognize character using smoothed character image
        recognized_char = recognize_character(smoothed_test_char, train_letters, priors, dynamic_noise_level)

        recognized_text += recognized_char
    return recognized_text


###################################################### IMPLEMENTING HMM #########################################################################

def calculate_initial_state_probabilities(train_txt_fname, space_bias=0.5):
    initial_counts = {char: 0 for char in TRAIN_LETTERS}
    total_starts = 0

    clean_text = parse_training_data(train_txt_fname)

    for word in clean_text.split():
        if word[0] in TRAIN_LETTERS:
            initial_counts[word[0]] += 1
            total_starts += 1

    for char in initial_counts:
        if char == ' ':
            initial_counts[char] *= space_bias
    # Calculate initial state probabilities
    initial_probabilities = {char: (initial_counts[char] / total_starts if total_starts > 0 else 0) for char in TRAIN_LETTERS}
    
    return initial_probabilities


def calculate_transition_probabilities(train_txt_fname, space_bias=0.5):
    laplace_constant = 1  # This is the small value added to each count
    num_states = len(TRAIN_LETTERS)  # Number of states (characters)

    # Initialize transition counts with the Laplace constant
    transition_counts = {char: {next_char: laplace_constant for next_char in TRAIN_LETTERS} for char in TRAIN_LETTERS}
    total_transitions = {char: laplace_constant * num_states for char in TRAIN_LETTERS}

    # Parse the training text
    clean_text = parse_training_data(train_txt_fname)
    
    # Apply bias to transitions leading to a space
    for char in transition_counts:
        transition_counts[char][' '] *= space_bias
        total_transitions[char] += (laplace_constant * space_bias - laplace_constant)

    
    # Calculate transition counts using the clean text
    for i in range(len(clean_text) - 1):
        if clean_text[i] in TRAIN_LETTERS and clean_text[i+1] in TRAIN_LETTERS:
            transition_counts[clean_text[i]][clean_text[i+1]] += 1
            total_transitions[clean_text[i]] += 1

    # Calculating transition probabilities with Laplace smoothing
    transition_probabilities = {char: {next_char: (transition_counts[char][next_char] / total_transitions[char]) 
                                       for next_char in TRAIN_LETTERS} for char in TRAIN_LETTERS}
    
    return transition_probabilities

def safe_log(x):
    return np.log(x) if x > 0 else -np.inf

def viterbi_algorithm(test_letters, train_letters, transition_probabilities, initial_state_probabilities, priors, average_images, space_noise_increase_factor=1.5):
    num_states = len(TRAIN_LETTERS)
    num_observations = len(test_letters)

    log_zero = -np.inf
    viterbi = np.full((num_states, num_observations), log_zero)
    backpointer = np.zeros((num_states, num_observations), dtype=int)

    for state in range(num_states):
        char_key = TRAIN_LETTERS[state]
        train_char = train_letters[char_key]
        dynamic_noise_level = estimate_noise_level(test_letters[0], average_images)
        viterbi[state, 0] = safe_log(initial_state_probabilities[char_key]) + safe_log(calculate_likelihoods(test_letters[0], train_char, dynamic_noise_level, char_key, space_noise_increase_factor))
    
    for obs in range(1, num_observations):
        for state in range(num_states):
            char_key = TRAIN_LETTERS[state]
            train_char = train_letters[char_key]
            dynamic_noise_level = estimate_noise_level(test_letters[obs], average_images)
            emission_log_prob = safe_log(calculate_likelihoods(test_letters[obs], train_char, dynamic_noise_level, char_key, space_noise_increase_factor))

            log_probs = viterbi[:, obs - 1] + np.array([safe_log(transition_probabilities[TRAIN_LETTERS[prev_state]][char_key]) for prev_state in range(num_states)]) + emission_log_prob

            max_log_prob = np.max(log_probs)
            prev_state = np.argmax(log_probs)
            viterbi[state, obs] = max_log_prob
            backpointer[state, obs] = prev_state

    most_likely_sequence = []
    last_state = np.argmax(viterbi[:, num_observations - 1])
    most_likely_sequence.append(TRAIN_LETTERS[last_state])

    for obs in range(num_observations - 1, 0, -1):
        last_state = backpointer[last_state, obs]
        most_likely_sequence.insert(0, TRAIN_LETTERS[last_state])

    return ''.join(most_likely_sequence)


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
def array_to_string(array):
    return "\n".join(["".join(['*' if pixel else ' ' for pixel in row]) for row in array])

#####
def main():
    if len(sys.argv) != 4:
        raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

    (train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]

    # Load the training and test letters
    train_letters = load_training_letters(train_img_fname)
    test_letters = load_letters(test_img_fname)
    
    # After loading training letters
    average_images = calculate_average_images(train_letters)
    
    # Calculate the prior probabilities from the training text
    priors = calculate_priors(train_txt_fname)

     # Recognize text from the test image using Simple Bayes Net
    simple_recognized_text = ""
    for test_char in test_letters:
        dynamic_noise_level = estimate_noise_level(test_char, average_images)
        recognized_char = recognize_character(test_char, train_letters, priors, dynamic_noise_level, space_noise_increase_factor=1.5)
        simple_recognized_text += recognized_char

    print("Simple:", simple_recognized_text)
    
    # Calculate transition and initial state probabilities
    transition_probabilities = calculate_transition_probabilities(train_txt_fname)
    initial_state_probabilities = calculate_initial_state_probabilities(train_txt_fname)

    # Recognize text using the Viterbi algorithm for HMM
    hmm_recognized_text = viterbi_algorithm(test_letters, train_letters, transition_probabilities, initial_state_probabilities, priors, average_images)
    print("   HMM:", hmm_recognized_text)
    
    # print("testing")
    
if __name__ == "__main__":
    main()






# RESOURCES
# https://chat.openai.com/