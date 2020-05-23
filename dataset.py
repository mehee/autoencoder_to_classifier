import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

class Dataset:
    def __init__(self, DATASET_NAME):
        self.DATASET_NAME = DATASET_NAME

        (self.raw_train, self.raw_validation, self.raw_test), self.metadata = tfds.load(
        self.DATASET_NAME,
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True, 
        )

    def set_img_size(self, IMG_SIZE):
        self.IMG_SIZE = IMG_SIZE

    def format_example(self, image, label):
        image = tf.cast(image, tf.float32)
        image = (image/255.0) 
        image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        return image, label

    def format_data(self):
        train = self.raw_train.map(self.format_example)
        validation = self.raw_validation.map(self.format_example)
        test = self.raw_test.map(self.format_example)
        return train, validation, test
    

    def get_batches(self, TRAIN_SET, VALIDATION_SET, TEST_SET, BATCH_SIZE, SHUFFLE_BUFFER_SIZE):
        train_batches = TRAIN_SET.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        validation_batches = VALIDATION_SET.batch(BATCH_SIZE)
        test_batches = TEST_SET.batch(BATCH_SIZE)
        return train_batches, validation_batches, test_batches