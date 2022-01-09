import os
import sys
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
# from tensorflow.contrib.eager.python import tfe
from tensorflow.python.keras.datasets import mnist, fashion_mnist, cifar10, cifar100


def _compute_preds_loss_grad(model, x, y):
    """
    Computes the loss and gradients of a trainable model.

    Args:
        model: A tf.keras.Model
        x: training data
        y: ground truth labels

    Returns:
        a tuple (preds, loss, gradients) of lists.
    """
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = tf.keras.losses.categorical_crossentropy(y, y_pred)
    
    gradients = tape.gradient(loss, model.variables)
    return y_pred, loss, gradients


def train_base(model_fn, dataset, epochs=300, batchsize=32, lr=1e-3, model_name=None, device=None):
    """
    Trains the base classifier (aka the attacked classifier).

    Trains only Neural Networks.

    Args:
        model_fn: A callable function that returns a tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset: Name of the dataset as a string.
        epochs: Number of epochs to train the model.
        batchsize: Size of each batch.
        lr: Initial learning rate.
        model_name: Name of the model being built.
        device: Device to place the model on.

    """
    if device is None:
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'
    
    # Load the dataset
    (X_train, y_train), (X_test, y_test) = generic_utils.load_dataset(dataset)
    
    num_classes = y_train.shape[-1]
    num_train_batches = X_train.shape[0] // batchsize + int(X_train.shape[0] % batchsize != 0)
    num_test_batches = X_test.shape[0] // batchsize + int(X_test.shape[0] % batchsize != 0)
    
    # build the datasets
    train_dataset, test_dataset = generic_utils.prepare_dataset(X_train, y_train,
                                                                X_test, y_test,
                                                                batch_size=batchsize,
                                                                device=device)
    
    # construct the model on the correct device
    with tf.device(device):
        if model_name is not None:
            model = model_fn(num_classes, name=model_name)  # type: tf.keras.Model
        else:
            model = model_fn(num_classes)  # type: tf.keras.Model
    
    lr_schedule = tf.train.exponential_decay(lr, tf.train.get_or_create_global_step(),
                                             decay_steps=num_train_batches, decay_rate=0.99,
                                             staircase=True)
    
    optimizer = tf.train.AdamOptimizer(lr_schedule)
    
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer,
                                     global_step=tf.train.get_or_create_global_step())
    
    model_name = model.name if model_name is None else model_name
    basepath = 'weights/%s/%s/' % (dataset, model_name)
    
    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)
    
    checkpoint_path = basepath + model_name
    
    best_loss = np.inf
    
    print()
    
    # train loop
    for epoch_id in range(epochs):
        train_loss = tfe.metrics.Mean()
        test_loss = tfe.metrics.Mean()
        
        train_acc = tfe.metrics.Mean()
        test_acc = tfe.metrics.Mean()
        
        # train
        with tqdm(train_dataset,
                  desc="Epoch %d / %d: " % (epoch_id + 1, epochs),
                  total=num_train_batches, unit=' samples') as iterator:
            
            for train_iter, (x, y) in enumerate(iterator):
                y_preds, loss_vals, grads = _compute_preds_loss_grad(model, x, y)
                loss_val = tf.reduce_mean(loss_vals)
                
                # update model weights
                grad_vars = zip(grads, model.variables)
                optimizer.apply_gradients(grad_vars, tf.train.get_or_create_global_step())
                
                # compute and update training target_accuracy
                acc_val = tf.keras.metrics.categorical_accuracy(y, y_preds)
                
                train_loss(loss_val)
                train_acc(acc_val)
                
                if train_iter >= num_train_batches:
                    break
        # test
        with tqdm(test_dataset, desc='Evaluating',
                  total=num_test_batches, unit=' samples') as iterator:
            for x, y in iterator:
                y_preds = model(x, training=False)
                loss_val = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, y_preds))
                
                # compute and update the test target_accuracy
                acc_val = tf.keras.metrics.categorical_accuracy(y, y_preds)
                
                test_loss(loss_val)
                test_acc(acc_val)
        
        print("\nEpoch %d: Train Loss = %0.5f | Train Acc = %0.6f | Test Loss = %0.5f | Test Acc = %0.6f" % (
            epoch_id + 1, train_loss.result(), train_acc.result(), test_loss.result(), test_acc.result()
        ))
        
        train_loss_val = train_loss.result()
        if best_loss > train_loss_val:
            print("Saving weights as training loss improved from %0.5f to %0.5f!" % (best_loss, train_loss_val))
            print()
            
            best_loss = train_loss_val
            
            checkpoint.write(checkpoint_path)
    
    print("\n\n")
    print("Finished training !")


def evaluate_model(model_fn, dataset_name, batchsize=128, model_name=None, device=None):
    """
    Evaluates the base classifier (aka the attacked classifier).

    Evaluates only Neural Networks.

    Args:
        model_fn: A callable function that returns a tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        batchsize: Size of each batch.
        model_name: Name of the model being built.
        device: Device to place the model on.

    Returns:
        (test_loss, test_acc)
    """
    if device is None:
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'
    
    # Load the dataset
    (X_train, y_train), (X_test, y_test) = generic_utils.load_dataset(dataset_name)
    num_classes = y_train.shape[-1]
    
    num_test_batches = X_test.shape[0] // batchsize + int(X_test.shape[0] % batchsize != 0)
    
    # build the datasets
    train_dataset, test_dataset = generic_utils.prepare_dataset(X_train, y_train,
                                                                X_test, y_test,
                                                                batch_size=batchsize,
                                                                device=device)
    
    # construct the model on the correct device
    with tf.device(device):
        if model_name is not None:
            model = model_fn(num_classes, name=model_name)  # type: tf.keras.Model
        else:
            model = model_fn(num_classes)  # type: tf.keras.Model
    
    optimizer = tf.train.AdamOptimizer()
    
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer,
                                     global_step=tf.train.get_or_create_global_step())
    
    model_name = model.name
    basepath = 'weights/%s/%s/' % (dataset_name, model_name)
    
    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)
    
    checkpoint_path = basepath + model_name
    
    # restore the parameters that were saved
    checkpoint.restore(checkpoint_path)  # should be modified to read???
    
    # train loop
    test_loss = tfe.metrics.Mean()
    test_acc = tfe.metrics.Mean()
    
    with tqdm(test_dataset, desc='Evaluating',
              total=num_test_batches, unit=' samples') as iterator:
        for x, y in iterator:
            y_preds = model(x, training=False)
            loss_val = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, y_preds))
            
            # compute and update the test target_accuracy
            acc_val = tf.keras.metrics.categorical_accuracy(y, y_preds)
            
            test_loss(loss_val)
            test_acc(acc_val)
    print("\nTest Loss = %0.5f | Test Acc = %0.6f" % (test_loss.result(), test_acc.result()))
    
    return test_loss.result(), test_acc.result()


def train_classical_model(model_fn, dataset_name, model_name=None, evaluate=True):
    """
    Trains the base classifier (aka the attacked classifier).

    Trains only subclasses of BaseClassicalModel.

    Args:
        model_fn: A callable function that returns a subclassed BaseClassicalModel.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        model_name: Name of the model being built.
        evaluate: Whether to evaluate on the test set after training.
            This is only for observation, and takes significant time
            so it should be avoided unless absolutely necessary.
    """
    # Load the dataset
    (X_train, y_train), (X_test, y_test) = generic_utils.load_dataset(dataset_name)
    num_classes = y_train.shape[-1]
    
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    
    y_train = np.argmax(y_train, axis=-1)
    y_test = np.argmax(y_test, axis=-1)
    
    # construct the model on the correct device
    if model_name is not None:
        model = model_fn(num_classes, name=model_name)  # type: generic_utils.BaseClassicalModel
    else:
        model = model_fn(num_classes)  # type: generic_utils.BaseClassicalModel
    
    model_name = model.name if model_name is None else model_name
    basepath = 'weights/%s/%s/' % (dataset_name, model_name)
    
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    
    checkpoint_path = basepath + model_name + '.pkl'
    
    print()
    
    # train loop
    model.fit(X_train, y_train)
    
    # Save the model
    model.save(checkpoint_path)
    
    if evaluate:
        # Evaluate on train set once
        y_pred = model(X_train)
        
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=-1)
        
        train_accuracy = accuracy_score(y_train, y_pred)
        
        # Evaluate on test set once
        y_pred = model(X_test)
        
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=-1)
        
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print("\nTrain Acc = %0.6f" % (train_accuracy))
        print("Train Error = %0.6f" % (1. - train_accuracy))
        
        print("\nTest Acc = %0.6f" % (test_accuracy))
        print("Test Error = %0.6f" % (1. - test_accuracy))
        
        print("\n\n")
        print("Finished training !")


def evaluate_classical_model(model_fn, dataset_name, model_name=None):
    """
    Trains the base classifier (aka the attacked classifier).

    Trains only subclasses of BaseClassicalModel.

    Args:
        model_fn: A callable function that returns a subclassed BaseClassicalModel.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        model_name: Name of the model being built.

    Returns:
        (train_accuracy, test_accuracy)
    """
    # Load the dataset
    (X_train, y_train), (X_test, y_test) = generic_utils.load_dataset(dataset_name)
    num_classes = y_train.shape[-1]
    
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    
    y_train = np.argmax(y_train, axis=-1)
    y_test = np.argmax(y_test, axis=-1)
    
    # construct the model on the correct device
    if model_name is not None:
        model = model_fn(num_classes, name=model_name)  # type: generic_utils.BaseClassicalModel
    else:
        model = model_fn(num_classes)  # type: generic_utils.BaseClassicalModel
    
    model_name = model.name
    basepath = 'weights/%s/%s/' % (dataset_name, model_name)
    
    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)
    
    checkpoint_path = basepath + model_name + '.pkl'
    
    # restore the parameters that were saved
    model = model.restore(checkpoint_path)
    
    # Evaluate on train set once
    y_pred = model(X_train)
    
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    
    train_accuracy = accuracy_score(y_train, y_pred)
    
    # Evaluate on test set once
    y_pred = model(X_test)
    
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print("\nTrain Acc = %0.6f" % (train_accuracy))
    print("Train Error = %0.6f" % (1. - train_accuracy))
    
    print("\nTest Acc = %0.6f" % (test_accuracy))
    print("Test Error = %0.6f" % (1. - test_accuracy))
    
    return (train_accuracy, test_accuracy)


class BaseClassicalModel(object):
    
    def __init__(self, name, **kwargs):
        """
        Base class of all Classical models to provide a unified interface
        for the training and evaluation engines.

        Args:
            name: Name of the classifier.
        """
        if name is None:
            name = self.__class__.__name__
        
        self.name = name
    
    def fit(self, X, y, training=True, **kwargs):
        """
        Unified method to train the classifer. Has access to mode parameter -
        `training` usually used by Neural Networks, to be used if required.

        Args:
            X: Training dataset.
            y: Training labels.
            training: Bool flag, whether training mode or
                evaluation mode. To be ignored by downstream
                subclasses (unless its a Neural Network under
                black-box treatment).
        """
        raise NotImplementedError()
    
    def predict(self, X, training=False, **kwargs):
        """
        Unified method to evaluate the classifier. Has access to mode parameter -
        `training` usually used by Neural Networks, to be used if required.

        Args:
            X: Test dataset.
            training: Bool flag, whether training mode or
                evaluation mode. To be ignored by downstream
                subclasses (unless its a Neural Network under
                black-box treatment).

        Returns:
            Predictions of the model.
        """
        raise NotImplementedError()
    
    def __call__(self, *args, **kwargs):
        """ Dispatch to the call(.) function """
        return self.call(*args, **kwargs)
    
    def call(self, x, training=False, **kwargs):
        """
        Perform evaluation when the classifier is called by the engine.
        Equivalent to `predict` with care taken to dispatch to numpy
        and reshape to 2d tensor.

        This is preferred when evaluation is required.

        Args:
            x: Dataset samples.
            training: Bool flag, whether training mode or
                evaluation mode. To be ignored by downstream
                subclasses (unless its a Neural Network under
                black-box treatment).

        Returns:
            Predictions of the model.
        """
        if hasattr(x, 'numpy'):  # is a tensor input
            x = x.numpy()  # make it a numpy ndarray
        
        if x.ndim > 2:  # reshape it to a 2d matrix for classical models
            x = x.reshape((x.shape[0], -1))
        
        return self.predict(x, training, **kwargs)
    
    def save(self, filepath):
        """ Save the class and its self contained dictionary of values """
        state = (self.__class__, self.__dict__)
        joblib.dump(state, filepath)
    
    @classmethod
    def restore(cls, filepath):
        """ Restore the class and its self contained dictionary of values """
        if os.path.exists(filepath):
            state = joblib.load(filepath)
            obj = cls.__new__(state[0])
            obj.__dict__.update(state[1])
            return obj
        else:
            raise FileNotFoundError("Model not found at %s" % filepath)


# Obtained from keras.utils folder. Copied to remove unnecessary keras import.
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def checked_argmax(y, to_numpy=False):
    """
    Performs an argmax after checking if the input is either a tensor
    or a numpy matrix of rank 2 at least.

    Should be used in most cases for conformity throughout the
    codebase.

    Args:
        y: an numpy array or tensorflow tensor
        to_numpy: bool, flag to convert a tensor to a numpy array.

    Returns:
        an argmaxed array if possible, otherwise original array.
    """
    if hasattr(y, 'numpy'):
        if len(y.shape) > 1:
            y = tf.argmax(y, axis=-1)
        
        if to_numpy:
            return y.numpy()
        else:
            return y
    else:
        if y.ndim > 1:
            y = np.argmax(y, axis=-1)
        
        return y


def reranking(y, target, alpha):
    """
    Scales the activation of the target class, then normalizes to
    a probability distribution again.

    Args:
        y: The predicted label matrix of shape [N, C]
        target: integer id for selection of target class
        alpha: scaling factor for target class activations.
            Must be greater than 1.

    Returns:

    """
    # assert alpha > 1, "Alpha must be greater than 1."
    
    max_y = tf.reduce_max(y, axis=-1).numpy()
    
    if hasattr(y, 'numpy'):
        weighted_y = y.numpy()
    else:
        weighted_y = y
    
    weighted_y[:, target] = alpha * max_y
    
    weighted_y = tf.convert_to_tensor(weighted_y)
    
    result = weighted_y
    result = result / tf.reduce_sum(result, axis=-1, keepdims=True)  # normalize to probability distribution
    
    # print('** y normed ** ', result.shape, (np.mean(result.numpy()[:, target], )))
    
    return result


def rescaled_softmax(y, num_classes, tau=1.):
    """
    Scales the probability distribution of the input matrix by tau,
    prior to softmax being applied.

    Args:
        y: tensor / matrix of shape [N, C] or [N]
        num_classes: int, number of classes.
        tau: scaling temperature

    Returns:
        a scaled matrix of shape [N, C]
    """
    tau = float(tau)
    is_tensor = hasattr(y, 'numpy')
    
    if len(y.shape) > 1:
        # we are dealing with class probabilities of shape [N, C]
        y = tf.nn.softmax(y / tau, axis=-1)
    else:
        # we are dealing with class labels of shape [N], not class probabilities.
        # one hot encode the score and then scale.
        y = tf.one_hot(y, depth=num_classes)
        # y = tf.nn.softmax(y, axis=-1)
    
    if is_tensor:
        return y
    else:
        return y.numpy()


def target_accuracy(y_label, y_pred, target):
    """
    Computes the accuracy as well as num_adv of attack of the target class.

    Args:
        y_label: ground truth labels. Accepts one hot encodings or labels.
        y_pred: predicted labels. Accepts probabilities or labels.
        target: target class

    Returns:
        accuracy, target_rate
    """
    ground = checked_argmax(y_label, to_numpy=True)  # tf.argmax(y_label, axis=-1).numpy()
    predicted = checked_argmax(y_pred, to_numpy=True)  # tf.argmax(y_pred, axis=-1).numpy()
    accuracy = np.mean(np.equal(ground, predicted))
    
    non_target_idx = (ground != target)
    target_total = np.sum((predicted[non_target_idx] == target))
    target_rate = target_total / np.sum(non_target_idx)
    
    # Cases where non_target_idx is 0, so target_rate becomes nan
    if np.isnan(target_rate):
        target_rate = 1.  # 100% target num_adv for this batch
    
    return accuracy, target_rate


def enable_printing():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def disable_printing():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def split_dataset(x_test, y_test, test_fraction=0.5):
    """
    Accepts an inputs dataset, and selects a portion of the test set to be
    the new train set for the adversarial model.

    Tries to extract such that number of samples in the new train set is
    same as the number of samples in the original train set.

    Uses class wise splitting to maintain counts from the test set.

    Args:
        X_test: numpy array
        y_test: numpy array

    Returns:
        (X_train, y_train), (X_test, y_test) with reduced number of samples
    """
    np.random.seed(0)
    num_classes = y_test.shape[-1]
    
    y_test = checked_argmax(y_test)
    
    test_labels, test_counts = np.unique(y_test, return_counts=True)
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    
    # Split the test set into adversarial train and adversarial test splits
    for label, max_cnt in zip(test_labels, test_counts):
        samples = x_test[y_test.flatten() == label]
        train_samples, test_samples = train_test_split(samples, test_size=test_fraction, random_state=0)
        
        train_cnt = len(train_samples)
        max_cnt = train_cnt
        
        X_train.append(train_samples[:max_cnt])
        Y_train.append([label] * max_cnt)
        
        X_test.append(test_samples)
        Y_test.append([label] * len(test_samples))
    
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    
    from keras.utils import to_categorical
    y_train = to_categorical(np.concatenate(Y_train), num_classes)
    y_test = to_categorical(np.concatenate(Y_test), num_classes)
    
    print("\nSplitting test set into new train and test set !")
    print("X train = ", X_train.shape, "Y train : ", y_train.shape)
    print("X test = ", X_test.shape, "Y test : ", y_test.shape)
    
    return (X_train, y_train), (X_test, y_test)


def load_dataset(dataset_name='mnist'):
    """
    Resolves the provided dataset name into an image dataset provided
    by Keras dataset utils, or any ucr dataset (by id or name).

    Args:
        dataset_name: must be a string. Can be the dataset name
            for image datasets, and must be in the format
            'ucr/{id}' or 'ucr/{dataset_name}' for the ucr datasets.

    Returns:
        Image dataset : (X_train, y_train), (X_test, y_test)
        Time Series dataset : (X_train, y_train), (X_test, y_test), dictionary of dataset info
    """
    allowed_image_names = ['mnist', 'cifar10', 'cifar100', 'fmnist']
    
    if dataset_name in allowed_image_names:
        return load_image_dataset(dataset_name)
    
    ucr_split = dataset_name.split('/')
    if len(ucr_split) > 1 and ucr_split[0].lower() == 'ucr':
        # is a ucr dataset time series dataset
        id = -1
        
        try:
            id = int(ucr_split[-1])
        except ValueError:
            # assume it is a name of the time series dataset
            
            try:
                id = ucr_utils.DATASET_NAMES.index(ucr_split[-1].lower())
            except ValueError:
                print("Could not match %s to either id or name of dataset !" % (ucr_split[-1]))
        
        if id < 0:
            raise ValueError('Could not match %s to either id or name of dataset !' % (ucr_split[-1]))
        
        return load_ucr_dataset(id, normalize_timeseries=True)
    
    else:
        raise ValueError("Could not parse the provided dataset name : ", dataset_name)


def load_image_dataset(dataset_name='mnist'):
    """
    Loads the dataset by name.

    Args:
        dataset_name: string, either "mnist", "cifar10", "cifar100", "fmnist"

    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    
    allowed_names = ['mnist', 'cifar10', 'cifar100', 'fmnist']
    
    if dataset_name not in allowed_names:
        raise ValueError("Dataset name provided is wrong. Must be one of ", allowed_names)
    
    #  print(directory)
    if dataset_name == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == 'fmnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    elif dataset_name == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    elif dataset_name == 'cifar100':
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    else:
        raise ValueError('%s is not a valid dataset name. Available choices are : %s' % (
            dataset_name, str(allowed_names)
        ))
    
    if dataset_name in ['mnist', 'fmnist']:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.
    
    elif dataset_name in ['cifar10', 'cifar100']:
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.
    
    if dataset_name == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10
    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    return (X_train, y_train), (X_test, y_test)


def prepare_dataset(X_train, y_train, X_test, y_test, batch_size, shuffle=True, device=None):
    """
    Constucts a train and test tf.Dataset for usage.

    Shuffles, repeats and batches the train set. Only batches the test set.
    Both datasets are pushed to the correct device for faster processing.

    Args:
        X_train: Train data
        y_train: Train label
        X_test: Test data
        y_test: Test label
        batch_size: batch size of the dataset
        shuffle: Whether to shuffle the train dataset or not
        device: string, 'cpu:0' or 'gpu:0' or some variant of that sort.

    Returns:
        two tf.Datasets, train and test datasets.
    """
    if device is None:
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    
    if shuffle:
        train_dataset = train_dataset.apply(tf.data.experimental.shuffle_and_repeat(1000, seed=0))
    
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.apply(tf.data.experimental.prefetch_to_device(device))
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.apply(tf.data.experimental.prefetch_to_device(device))
    
    return train_dataset, test_dataset


def plot_image_adversary(sequence, title, ax, remove_axisgrid=False,
                         xlabel=None, ylabel=None, legend=False,
                         imlabel=None, color=None, alpha=1.0):
    """
    Utility method for plotting a sequence.

    Args:
        sequence: A time series sequence.
        title: Title of the plot.
        ax: Axis of the subplot.
        remove_axisgrid: Whether to remove the axis grid.
        xlabel: Label of X axis.
        ylabel: Label of Y axis.
        legend: Whether to enable the legend.
        imlabel: Whether to label the sequence (for legend).
        color: Whetehr the sequence should be of certain color.
        alpha: Alpha value of the sequence.

    Returns:
        Note, this method does not automatically call plt.show().

        This is to allow multiple subplots to borrow the same call
        without immediate plotting.

        Therefore, do not forget to call `plt.show()` at the end.
    """
    if remove_axisgrid:
        ax.axis('off')
    
    if sequence.ndim > 3:
        raise ValueError("Data provided cannot be more than rank 3 tensor.")
    else:
        ax.plot(sequence.flatten(), color=color, alpha=alpha, label=imlabel)
        
        if title is not None:
            ax.set_title(str(title), fontsize=20)
    
    if xlabel is not None:
        ax.get_xaxis().set_visible(True)
        ax.set_xlabel(str(xlabel))
    
    if ylabel is not None:
        ax.get_yaxis().set_visible(True)
        ax.set_ylabel(str(ylabel))
    
    if legend:
        ax.legend(loc='upper right')
