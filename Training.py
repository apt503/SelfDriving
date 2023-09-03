import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import GlobalAveragePooling1D, BatchNormalization, ConvLSTM2D, Reshape, Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Dropout, TimeDistributed, LSTM, Flatten, Input, Dense, GlobalAveragePooling2D
from keras.utils import Sequence
from keras.applications import MobileNetV2
from keras.applications import ResNet50
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import backend as K
import random
import csv
import os

rows, cols, dimensions = 244, 244, 3
batch_size = 32


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size=batch_size, dim=(rows, cols), n_channels=dimensions, n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 1), dtype=float)

        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.load('/CNN/ProcessedData/data/' + str(ID) + '.npy')
            y[i] = np.load('/CNN/ProcessedData/labels/' + str(ID) + '.npy')

        return X, y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

class MetricsLoggerCallback(Callback):
    def __init__(self, log_file):
        super(MetricsLoggerCallback, self).__init__()
        self.log_file = log_file

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1}:\n")
            f.write(f"Training Loss: {train_loss}\n")
            f.write(f"Validation Loss: {val_loss}\n")


def load_image_as_np_array(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)  # change this to the appropriate function if not a JPEG
    return img.numpy()


def load_label_as_np_array(filename):
    csv_dataset = tf.data.experimental.CsvDataset(filename, [tf.double, tf.double, tf.double])
    label = np.array(list(csv_dataset))
    return label


def simple_resize(image, target_height, target_width):
    height, width, channels = image.shape
    x_indices = np.linspace(0, width - 1, target_width, dtype=int)
    y_indices = np.linspace(0, height - 1, target_height, dtype=int)
    return image[np.ix_(y_indices, x_indices)]


def weighted_mse(y_true, y_pred):
    # Calculate the absolute values of y_true
    # This gives higher weights to samples with labels further from 0
    weights = K.abs(y_true) + 0.1

    # Add a small value to avoid division by zero
    weights = weights + K.epsilon()

    # Calculate the squared error
    error = K.square(y_pred - y_true)

    # Return the mean of the weighted error
    return K.mean(weights * error)

def PreProcess():
    image_dir = '/CNN/data/img_data/'
    label_dir = '/CNN/data/pwm_data/'

    image_files = sorted(os.listdir(image_dir), key=lambda x: int(x.split('.')[0]))
    label_files = sorted(os.listdir(label_dir), key=lambda x: int(x.split('.')[0]))

    folder_name = "/CNN/ProcessedData/"
    input_path = os.path.join(folder_name, 'data/')
    label_path = os.path.join(folder_name, 'labels/')

    if not os.path.exists(input_path):
        os.makedirs(input_path)
        os.makedirs(label_path)

    list_ids = []

    num = 1
    # we start from the second image and stop at the second-to-last
    for i in range(1, len(image_files)):

        label = load_label_as_np_array(label_dir + label_files[i])

        if -0.25 < label[0,0] < 0.25:
            image_array = simple_resize(load_image_as_np_array(image_dir + image_files[i]), 244, 244)

            image_array = np.array(image_array).astype('float32') / 255.0
            #image_array = np.transpose(image_array, (2, 1, 3))
            np.save(input_path + str(num) + '.npy', image_array)

            label = load_label_as_np_array(label_dir + label_files[i])

            np.save(label_path + str(num) + '.npy', label[0, 0])

            list_ids.append(num)
            num += 1

    return list_ids


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    #ids = PreProcess()
    ids = list(range(1, 46253))


    removed_ids = []


    potential_start_points = set(ids[:-200])  # Create a set of potential start points

    for _ in range(20):
        start_point = random.choice(list(potential_start_points))

        # Remove this start point and its range from potential start points to avoid overlaps
        for i in range(start_point, start_point + 200):
            potential_start_points.discard(i)
            if i in ids:
                ids.remove(i)
                removed_ids.append(i)

    # Writing the removed_ids to a file
    with open("/CNN/removed_ids.txt", "w") as file:
        for r_id in removed_ids:
            file.write(str(r_id) + '\n')

    random.shuffle(ids)

    training_generator = DataGenerator(ids[:int(len(ids) * 0.8)])  
    validation_generator = DataGenerator(ids[int(len(ids) * 0.8):])

    test_generator = DataGenerator(removed_ids)
    image_input = Input(shape=(rows, cols, dimensions))
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(rows, cols, dimensions))
    CNN_features = base_model(image_input)
    global_features = GlobalAveragePooling2D()(CNN_features)

    x = Dense(256, activation='relu')(global_features)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='linear')(x)

    model = Model(inputs=image_input, outputs=predictions)
    checkpoint_path = "/CNN/LowComplexityNetwork.h5"
    checkpoint_callback = ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

    model.compile(optimizer='adam',
                  loss=weighted_mse,
                  metrics=['mae'])

    print(model.summary())

    output_file = "/CNN/epoch_predictions.txt"
    metrics_file = "/CNN/epoch_metrics.txt"

    model.fit(training_generator,
              validation_data=validation_generator,
              epochs=250,
              callbacks=[MetricsLoggerCallback(metrics_file),
                         checkpoint_callback,
                         early_stopping
                         ])

    model.save('/CNN/model.keras')
    model.save('/CNN/modelstore.h5')

    predictions = model.predict(test_generator)

    true_outputs = []
    for _, labels_batch in test_generator:
        true_outputs.extend(labels_batch)

    with open('/CNN/results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["True Output", "Predicted Output"])

        for true, predicted in zip(true_outputs, predictions):
            writer.writerow([true[0], predicted[0]])


    print("complete")