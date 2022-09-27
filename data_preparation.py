from numpy import array


class SplitSequence:
    __instance = None

    @staticmethod 
    def get_instance():
        if SplitSequence.__instance==None:
            SplitSequence()
        return SplitSequence.__instance
    def __init__(self):
  
        if SplitSequence.__instance != None:
            raise ValueError("You can not take more than one instance")
        else:
            SplitSequence.__instance = self

    def univariate(self, sequence, n_steps):

        """
        :models -> Vanilla LSTM
                -> Stacked LSTM
                -> Bidirectional LSTM
                -> CNN LSTM
                -> ConvLSTM

        :param sequence:
        :param n_steps:
        :return (numpy.array(X), numpy.array(y)):
        """
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence) - 1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    def multivariate(self, sequences, n_steps):
        """
        :models -> Multiple Input Series


        :param sequences:
        :param n_steps:
        :return (numpy.array(X), numpy.array(y)):
        """
        X, y = list(), list()
        for i in range(len(sequences)):
            end_ix = i + n_steps
            if end_ix > len(sequences):
                break
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    def multiple_parallel_series(self, sequences, n_steps):

        """
        models -> Multiple Parallel Series

        :param sequences:
        :param n_steps:
        :return (numpy.array(X), numpy.array(y)):
        """

        X, y = list(), list()
        for i in range(len(sequences)):
            end_ix = i + n_steps
            if end_ix > len(sequences) - 1:
                break
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    def multi_step(self, sequence, n_steps_in, n_steps_out):

        """
        models -> Vector Output Model
               -> Encoder-Decoder Model

        :param sequence:
        :param n_steps_in:
        :param n_steps_out:
        :return (numpy.array(X), numpy.array(y)):
        """

        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    def multivariate_multi_step(self, sequences, n_steps_in, n_steps_out):

        """
        models -> Multiple Input Multi-Step Output


        :param sequences:
        :param n_steps_in:
        :param n_steps_out:
        :return (numpy.array(X), numpy.array(y)):
        """
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out - 1
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1:out_end_ix, -1]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    def multiple_parallel_input_and_multi_step_output(self, sequences, n_steps_in, n_steps_out):
        """
        models -> Multiple Parallel Input and Multi-Step Output

        :param sequences:
        :param n_steps_in:
        :param n_steps_out:
        :return (numpy.array(X), numpy.array(y)):
        """
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)
