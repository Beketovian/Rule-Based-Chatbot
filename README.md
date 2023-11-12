# Rule-Based Chatbot

## Description
This is an example of an old Rule-Based chatbot for the Fall 2023 Aggie Coding Club Generative AI workshop.
## File Descriptions
- **intents.json**: Contains the data for training the chatbot, including user intents, patterns, and corresponding responses. These are responses based on any keywords found in the input.
- **training.py**: Used for training the chatbot model. Processes the data from `intents.json` to prepare the chatbot for interaction.
- **chatbot.py**: The main script for the chatbot. Handles user inputs and generates responses based on the trained model.

## Instructions
1. **Training the Model**: Start by running `training.py`. This script will train the chatbot model using the data in `intents.json`, setting up the necessary patterns for the chatbot to recognize.
2. **Running the Chatbot**: After training, execute `chatbot.py` to launch the chatbot. The chatbot will interact with users based on the trained patterns and responses.
