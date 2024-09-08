"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List
import scipy.linalg
import numpy as np

N_DIMENSIONS = 10


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray,k:int) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    # Compute distances
    distances = -2 * np.dot(test, train.T) + np.sum(train**2, axis=1) + np.sum(test**2, axis=1)[:, np.newaxis]
    # Find indices of k-nearest neighbors
    nearest = np.argsort(distances, axis=1)[:, :k]

    # Get labels of k-nearest neighbors
    label_k = train_labels[nearest]

    # Majority voting
    labels = []
    for row in label_k:
        unique_elements, counts = np.unique(row, return_counts=True)
        max_count = np.max(counts)
        candidates = [label for label, count in zip(unique_elements, counts) if count == max_count]
        selected_label = np.random.choice(candidates)  # Handle ties by random selection
        labels.append(selected_label)

    return labels





# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    
    if (model["storing_eginv"]==None):
            covx = np.cov(data, rowvar=0)
            N = covx.shape[0]
            w, v = scipy.linalg.eigh(covx, subset_by_index=(N - N_DIMENSIONS, N - 1))
            model["storing_eginv"]=v.tolist()
            pcatrain_data = np.dot((data - np.mean(data)), v)
    else:
        pcatrain_data = np.dot((data - np.mean(data)), model["storing_eginv"])
    return pcatrain_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.
    model = {}
    model["labels_train"] = labels_train.tolist()
    model["storing_eginv"] = None
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test,8)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    final_boards = []
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])
    
    # Convert fvectors_test to a 4D array
    square_number = 64
    board_size = 8
    test_boards_array = fvectors_test.reshape(-1, board_size, board_size, fvectors_test.shape[1])

    # Classify squares
    result_list = classify_squares(fvectors_test, model)
    
    # Reshape the predicted labels to a 3D array
    num_boards = len(result_list) // 64
    test_label = np.array(result_list).reshape((num_boards, board_size, board_size))

    # Count the number of pieces in each board
    counts_per_board = count_pieces(test_label)

    # Correct misclassifications
    correct_misclassifications(test_label, counts_per_board, test_boards_array, fvectors_train, labels_train)

    return test_label.flatten().tolist()
def count_pieces(test_label):
    # Count the number of pieces in each board
    counts_per_board = []
    for board in test_label:
        board_counts = {piece: 0 for piece in 'KQRNBPkqrbnp.'}
        for row in board:
            for piece in row:
                board_counts[piece] += 1
        counts_per_board.append(board_counts)
    return counts_per_board


def correct_misclassifications(test_label, counts_per_board, test_boards_array, fvectors_train, labels_train):
    # Correct misclassifications
    index_error = []
    fvector_error = []

    for counter, board_counts in enumerate(counts_per_board):
        for piece in 'KQRNBPkqrbnp':
            if board_counts[piece] > get_max_count(piece):
                index = list(zip(*np.where(test_label[counter] == piece)))
                for cord in index:
                    index_error.append((counter, cord[0], cord[1]))
                    fvector_error.append(test_boards_array[counter, cord[0], cord[1], :])

    fvector_error = np.array(fvector_error)

    # Reclassify with an increasing value of k
    k = 3
    while any(board_counts[piece] > get_max_count(piece) for board_counts in counts_per_board for piece in 'KQRNBPkqrbnp'):
        if k > 12:
            break
        labels = classify(fvectors_train, labels_train, fvector_error, k)
        for i, index in enumerate(index_error):
            test_label[index] = labels[i]
        k += 1


def get_max_count(piece):
    # Define the maximum allowed count for each piece
    max_counts = {'K': 1, 'Q': 1, 'R': 2, 'N': 2, 'B': 2, 'P': 8, 'k': 1, 'q': 1, 'r': 2, 'n': 2, 'b': 2, 'p': 8, '.': 64}
    return max_counts[piece]