
import gym
import numpy as np

def ReduceResolutionOFImage(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

def remove_color_from_Image(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background_from_Image(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image
def set_everything_else_to_one(image):
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1
    return image

def preprocess_observations_image(current_observation_image, previous_processed_observation_image, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation_image = current_observation_image[35:195] # crop
    processed_observation_image = ReduceResolutionOFImage(processed_observation_image)
    processed_observation_image = remove_color_from_Image(processed_observation_image)
    processed_observation_image = remove_background_from_Image(processed_observation_image)
    processed_observation_image = set_everything_else_to_one(processed_observation_image)
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation_image = processed_observation_image.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if previous_processed_observation_image is not None:
        current_observation_image = processed_observation_image - previous_processed_observation_image
    else:
        current_observation_image = np.zeros(input_dimensions)
    # update the previous observation_image so that we can use it to get the new changes after the next iteration
    processed_observation_image = processed_observation_image
    return current_observation_image, processed_observation_image


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector

def apply_neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values

def choose_action_up_OR_down(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # signifies up in pong implmentation of open ai gym
        return 2
    else:
         # signifies down in pong implmentation of open ai gym
        return 3

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    """ See here: http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop"""
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name]) # reset batch gradient buffer

def discount_rewards(rewards, gamma):
    """ Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago.
    This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    """ discount the gradient with the normalized rewards """
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards

    #Define the model in keras (WIP)
def keras_learning_model(input_dim=80*80, model_type=1):
  model = Sequential()
  if model_type==0:
    model.add(Reshape((1,80,80), input_shape=(input_dim,)))
    model.add(Flatten())
    model.add(Dense(200, activation = 'relu'))
    model.add(Dense(number_of_inputs, activation='softmax'))
    opt = RMSprop(lr=learning_rate)
  else:
    model.add(Reshape((1,80,80), input_shape=(input_dim,)))
    model.add(Convolution2D(32, 9, 9, subsample=(4, 4), border_mode='same', activation='relu', init='he_uniform'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu', init='he_uniform'))
    model.add(Dense(number_of_inputs, activation='softmax'))
    opt = Adam(lr=learning_rate)
  model.compile(loss='categorical_crossentropy', optimizer=opt)
  if resume == True:
    model.load_weights('pong_model_checkpoint.h5')
  return model

model = learning_model()


def start(): # where the execution starts
    env = gym.make("Pong-v0") # start the open ai pong environment
    observation_image = env.reset() # This gets us the image

    # hyperparameters
    episode_number = 0
    batch_size = 10
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80 * 80
    learning_rate = 0.001 #1e-4 : original value for RMSPROP , we are using adam optimizer here in keras model . Link : https://keras.io/optimizers/


    #Script Parameters for keras
    update_frequency = 1 # to decide how often to update the keras model parameters
    resume = False # to load a previous checkpoint model weights to run again.
    render = True # in order to render the open ai environment.
    train_X = []
    train_y = []

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], [] # to be used instead of the below line
    xs, dlogps, drs, probs = [],[],[],[] # keras script same values

    episode_number = 0
    reward_sum = 0
    running_reward = None
    previous_processed_observations_image_vector = None

    weights = {
        '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    }

    # To be used with rmsprop algorithm (http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])


    model = learning_model() # compiling the keras model.

    while True:
        env.render()
        processed_observations_image_vector, previous_processed_observations_image_vector = preprocess_observations_image(observation_image, previous_processed_observations_image_vector, input_dimensions)

        #Predict probabilities from the Keras model
        up_probability = ((model.predict(processed_observations_image_vector.reshape([1,processed_observations_image_vector.shape[0]]), batch_size=1).flatten()))

        # old code hidden_layer_values, up_probability = apply_neural_nets(processed_observations_image_vector, weights)

        episode_observations.append(processed_observations_image_vector)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action_up_OR_down(up_probability)

        # carry out the chosen action and get back the details from the enviroment after performing the action
        observation_image, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)

        # see here: http://cs231n.github.io/neural-networks-2/#losses
        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_gradient_log_ps.append(loss_function_gradient)


        if done: # an episode finished
            episode_number += 1

            # Combine the following values for the episode
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            # Tweak the gradient of the log_ps based on the discounted rewards
            episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma)

            gradient = compute_gradient(
              episode_gradient_log_ps_discounted,
              episode_hidden_layer_values,
              episode_observations,
              weights
            )

            # Sum the gradient for use when we hit the batch size
            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            if episode_number % batch_size == 0:
                update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)

            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], [] # reset values
            observation_image = env.reset() # reset env
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
            reward_sum = 0
            previous_processed_observations_image_vector = None

start()
