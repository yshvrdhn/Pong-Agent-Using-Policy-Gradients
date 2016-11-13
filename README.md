# Pong AI using Policy Gradient

### Description

The code in this repository will let you train a Convolutional Neural Network(CNN) to use OpenAI gym to play Pong solely using input frames of the game. The CNN is written in [Keras](https://github.com/fchollet/keras).
The code in `pong_keras.py` is based on Andrej Karpathy's blog on Deep Reinforcement Learning.
You can play around with other such Atari games at the [Openai Gym](https://gym.openai.com).

### Setup

1. Follow the instructions for installing Openai Gym [here](https://gym.openai.com/docs). You may need to install `cmake` first.
2. Run `pip install -e '.[atari]'`.
3. Run `python pong_keras.py`
