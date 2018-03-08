import sys
import time

from keras.models import *
from keras.layers import *
from keras.optimizers import *

sys.path.append('..')
from NeuralNet import NeuralNet
from utils import dotdict

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 12,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

class GeneralsNNet():
    def __init__(self, game, args):
        # game params
        self.width, self.height = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # four layers: troops, owns, generals, num steps away from 50
        self.input_boards = Input(shape=(self.width * self.height * 4,))
        x_image = Reshape((self.width, self.height, 4))(self.input_boards)

        # batch_size  x board_x x board_y x num_channels
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(x_image)))
         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv1)))
        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid')(h_conv2)))
        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid')(h_conv3)))
        h_conv4_flat = Flatten()(h_conv4)
        # batch_size x 1024
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))
        # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))
        # batch_size x self.action_size
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)
        # batch_size x 1
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = GeneralsNNet(game, args)
        self.width, self.height = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        list_boards, target_pis, target_vs = list(zip(*examples))

        input_boards = np.empty((len(examples), len(list_boards[0].to_array())))
        for i, board in enumerate(list_boards):
            input_boards[i] = board.to_array()

        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x = input_boards,
                            y = [target_pis, target_vs],
                            batch_size = args.batch_size,
                            epochs = args.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # run
        board_arr = np.expand_dims(board.to_array(), axis=0)
        pi, v = self.nnet.model.predict(board_arr)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)
