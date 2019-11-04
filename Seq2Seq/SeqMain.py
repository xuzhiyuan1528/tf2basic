import os
import time

import editdistance
import numpy as np
import tensorflow as tf
from Seq2Seq.Utils import ToyDataset


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden, **kwargs):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([toy.char_index(toy.SOS)] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def evaluate(dataset_eval):
    steps_per_epoch = len(input_train) // BATCH_SIZE
    acc = []

    for inp_eval, targ_eval in dataset_eval.take(steps_per_epoch):

        hidden = [tf.zeros((BATCH_SIZE, units))]
        enc_out, enc_hidden = encoder(inp_eval, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([toy.char_index(toy.SOS)] * BATCH_SIZE, 1)
        prediction = ['' for _ in range(BATCH_SIZE)]
        target = ['' for _ in range(BATCH_SIZE)]
        targ_eval = targ_eval.numpy()
        for t in range(toy.max_seq_len):
            predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                                 dec_hidden,
                                                                 enc_out)

            predicted_id = tf.argmax(predictions, axis=1).numpy()

            for i in range(BATCH_SIZE):
                prediction[i] += toy.index_char(predicted_id[i])
                if t != 0:
                    target[i] += toy.index_char(targ_eval[i][t])

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims(predicted_id, 1)

        for i in range(BATCH_SIZE):
            prediction[i] = prediction[i].split('/')[0]
            target[i] = target[i].split('/')[0]

        for pred, targ in zip(prediction, target):
            # print(pred, " : ", targ)
            acc.append(editdistance.eval(pred, targ))

    print("accuracy", 1. - np.mean(acc))


def translate(sentence):
    inputs = [toy.char_index(i) for i in sentence]
    inputs = [1] + inputs + [2]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=toy.max_seq_len,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([toy.char_index(toy.SOS)], 0)

    for t in range(toy.max_seq_len):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += toy.index_char(predicted_id) + ' '

        if toy.index_char(predicted_id) == toy.EOS:
            return result, sentence

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


if __name__ == '__main__':
    train_flag = False
    toy = ToyDataset(5, 10)
    inp_set, tar_set = toy.get_dataset(30000)
    input_train, input_eval, target_train, target_eval = toy.split_dataset(inp_set, tar_set, 0.2)

    BUFFER_SIZE = len(input_train)
    BATCH_SIZE = 32
    steps_per_epoch = len(input_train) // BATCH_SIZE
    embedding_dim = 256
    units = 1024
    vocab_inp_size = toy.vocab_size
    vocab_tar_size = toy.vocab_size

    dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    dataset_eval = tf.data.Dataset.from_tensor_slices((input_eval, target_eval)).shuffle(BUFFER_SIZE)
    dataset_eval = dataset_eval.batch(BATCH_SIZE, drop_remainder=True)

    example_input_batch, example_target_batch = next(iter(dataset))
    print(example_input_batch.shape, example_target_batch.shape)

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

    # sample input
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

    sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                          sample_hidden, sample_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

    optimizer = tf.keras.optimizers.Adam()

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    if train_flag:
        EPOCHS = 5
        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    result, sentence = translate('abccda')
    print(result, sentence)
    result, sentence = translate('aabbccd')
    print(result, sentence)

    evaluate(dataset_eval)

    # def translate(sentence):
    #     result, sentence, attention_plot = evaluate(sentence)
    #
    #     print('Input: %s' % (sentence))
    #     print('Predicted translation: {}'.format(result))
    #
    #     attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    #     # plot_attention(attention_plot, sentence.split(' '), result.split(' '))
    #
    #
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # translate(u'Hola.')
