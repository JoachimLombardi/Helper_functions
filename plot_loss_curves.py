def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  
  Args:
    history: TensorFlow History object.
  """

  import matplotlib.pyplot as plt
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  epochs = range(len(history.history["loss"])) # how many epochs did we run for ?

  # Plot loss

  plt.plot(epochs, loss, label="training_loss")
  plt.plot(epochs, val_loss, label="val_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  # Plot accuracy 
  plt.figure() # Separate the curves
  plt.plot(epochs, accuracy, label="training_accuracy")
  plt.plot(epochs, val_accuracy, label="val_accuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend()

# Visualize data
import matplotlib.pyplot as plt
import random
# Create a function for viewing images in a data batch
def show_batches_images(images, labels, augmented_images=None, augmented_label=None):
  """
  Displays a plot of random images and their labels from a data batch.
  """
  i = random.choice(range(len(images)))
  j = random.choice(range(len(images)))
  # Setup the figure
  plt.figure(figsize=(10, 10))
  # Create subplots (5 rows, 5 columns)
  plt.subplot(1, 2, 1)
  # Display an image
  plt.imshow(images[i])
  print("Showing image number:{}".format(i))
  # Add the image label as the title
  if labels.ndim == 1:
    plt.title(class_names[int(tf.round(labels[i]))])
  else:
    plt.title(class_names[labels[i].argmax()])
  # Turn the grid lines off
  plt.axis("off")
  # Create subplots (5 rows, 5 columns)
  plt.subplot(1, 2, 2)
  # Display an image
  plt.imshow(images[j])
  print("Showing image number:{}".format(j))
  # Add the image label as the title
  if labels.ndim == 1:
    plt.title(class_names[int(tf.round(labels[j]))])
  else:
    plt.title(class_names[labels[j].argmax()])
  # Turn the grid lines off
  plt.axis("off")
  if (augmented_images is not None):
    # Setup the figure
    plt.figure(figsize=(10, 10))
    # Create subplots (5 rows, 5 columns)
    plt.subplot(1, 2, 1)
    # Display an image
    plt.imshow(augmented_images[i])
    print("Showing augmented image number:{}".format(i))
    # Add the image label as the title
    if labels.ndim == 1:
      plt.title(class_names[int(tf.round(augmented_labels[i]))])
    else:
      plt.title(class_names[augmented_labels[i].argmax()])
      # Turn the grid lines off
      plt.axis("off")
      # Create subplots (5 rows, 5 columns)
      plt.subplot(1, 2, 2)
      # Display an image
      plt.imshow(augmented_images[j])
      print("Showing augmented image number:{}".format(j))
      # Add the image label as the title
    if labels.ndim == 1:
      plt.title(class_names[int(tf.round(augmented_labels[j]))])
    else:
      plt.title(class_names[augmented_labels[j].argmax()])
      # Turn the grid lines off
      plt.axis("off")
      
# Let's make a create_model() function to create a model from a URL
def create_model(dir_name, experiment_name, model_url=classifier_model, num_classes=10, loss="categorical_crossentropy", activation="softmax", epochs=5):
  """
  Takes a TensorFlow Hub URL and creates a keras Sequential model with it.

    Args:
      model_url (str): A TensorFlow Hub feature extraction URL.
      num_classes (int): Number of output neurons in the output layer.
      should be equal to number of target classes, default 10.
      dir_name (str): Name of the folder containing all tensorboard logs.
      experiment_name (str): Name of the folder containing tensorboard logs 
      corresponding to the pretrained model. 
      epochs (int): Number of iteration for all data.
      loss (str): Metrics for loss whether it is binary or multiclass .
      activation (str): Kind of outshape function whether it is binary or multiclass.
    
    Returns: 
      AN uncompiled Keras Sequential model with model_url as feature extractor 
      layer and Dense output layer with num_classes output neurons.
      A tensorflow history object.
  """
  # Download the pretrained model and save it as a keras layer
  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable=False, # freeze the already learned patterns
                                           name="feature_extraction_layer",
                                           input_shape=IMAGE_SHAPE+(3,))  
  
  # Create our own model
  model = tf.keras.Sequential([
      feature_extractor_layer, # Go through the pre-trained model first
      layers.Dense(num_classes, activation=activation, name="output_layer")
  ])

  # Compile our model
  model.compile(loss=loss,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
  
  # Fit our model (10 percent of 10 classes)
  tensorboard = create_tensorboard_callback(dir_name, experiment_name)
  history = model.fit(train_data,
                      epochs=epochs,
                      steps_per_epoch=len(train_data),
                      validation_data=test_data,
                      validation_steps=len(test_data),
                      callbacks=[tensorboard])

  return model, history

import os, datetime, pytz
# Create a function to save a model
def save_model(model, suffix=None):
  """
  Saves a given model in a models directory and appends a suffix (string).
  """
  # Create a model directory pathname with current time
  tz = pytz.timezone('Europe/Paris')
  modeldir = os.path.join("/content/drive/MyDrive/10_percent_food/save",
                          datetime.datetime.now().astimezone(tz).strftime("%d%m%Y-%H%M%S"))
  model_path = modeldir + "-" + suffix + ".h5" # Save format of model
  print(f"Saving model to: {model_path}...")
  model.save(model_path)
  return model_path

import tensorflow_hub as hub
# Create a function to load a trained model
def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                      custom_objects={"KerasLayer":hub.KerasLayer}) # first layer
  return model
