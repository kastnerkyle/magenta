from __future__ import print_function
import tensorflow as tf
import os
from tfkdllib import numpy_softmax, numpy_sample_softmax, piano_roll_to_midi
from tfkdllib import duration_and_pitch_to_midi


def validate_sample_args(model_ckpt,
                         prime,
                         sample,
                         sample_len,
                         temperature,
                         **kwargs):
    return (model_ckpt, prime, sample, sample_len, temperature)


def sample(kwargs):
    (model_ckpt,
     prime,
     sample,
     sample_len,
     temperature) = validate_sample_args(**kwargs)
    # Wow this is nastyyyyy
    from duration_rnn import *
    duration_mb, note_mb = train_itr.next()
    duration_and_pitch_to_midi("gt.mid", duration_mb[:, 0], note_mb[:, 0])
    train_itr.reset()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        model_dir = str(os.sep).join(model_ckpt.split(os.sep)[:-1])
        model_name = model_ckpt.split(os.sep)[-1]
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("Unable to restore from checkpoint")
        i_h1 = np.zeros((batch_size, h_dim)).astype("float32")
        i_h2 = np.zeros((batch_size, h_dim)).astype("float32")
        note_inputs = np.zeros((1, batch_size, train_itr.simultaneous_notes))
        duration_inputs = np.zeros((1, batch_size, train_itr.simultaneous_notes))
        note_targets = np.zeros((1, batch_size, train_itr.simultaneous_notes))
        duration_targets = np.zeros((1, batch_size, train_itr.simultaneous_notes))

        shp = note_inputs.shape
        full_notes = np.zeros((sample_len, shp[1], shp[2]), dtype="float32")
        full_notes[:len(note_inputs)] = note_inputs[:]
        shp = duration_inputs.shape
        full_durations = np.zeros((sample_len, shp[1], shp[2]), dtype="float32")
        full_durations[:len(duration_inputs)] = duration_inputs[:]

        random_state = np.random.RandomState(1999)
        for j in range(len(note_inputs[-1]), sample_len):
            # even predictions are note, odd are duration
            for ni in range(2 * n_notes):
                feed = {note_inpt: note_inputs,
                        note_target: note_targets,
                        duration_inpt: duration_inputs,
                        duration_target: duration_targets,
                        init_h1: i_h1,
                        init_h2: i_h2}
                outs = []
                outs += note_preds
                outs += duration_preds
                outs += [final_h1, final_h2]
                r = sess.run(outs, feed)
                h_l = r[-2:]
                h1_l, h2_l = h_l
                this_preds = r[:-2]
                this_probs = [numpy_softmax(p, temperature=temperature)
                              for p in this_preds]
                this_samples = [numpy_sample_softmax(p, random_state)
                                for p in this_probs]
                note_probs = this_probs[:n_notes]
                duration_probs = this_probs[n_notes:]
                si = ni // 2
                if (ni % 2) == 0:
                    # only put the single note in...
                    full_notes[j, :, si] = this_samples[si].ravel()
                    note_targets[0, :, si] = this_samples[si].ravel()
                if (ni % 2) == 1:
                    full_durations[j, :, si] = this_samples[si + n_notes].ravel()
                    duration_targets[0, :, si] = this_samples[si + n_notes].ravel()
            # priming sequence
            note_inputs = full_notes[j:j+1]
            duration_inputs = full_durations[j:j+1]
            note_targets = np.zeros((1, batch_size, train_itr.simultaneous_notes))
            duration_targets = np.zeros((1, batch_size, train_itr.simultaneous_notes))
            i_h1 = h1_l
            i_h2 = h2_l
            duration_and_pitch_to_midi("temp.mid", full_durations[:, 0], full_notes[:, 0])


if __name__ == '__main__':
    # prime is the text to prime with
    # sample is 0 for argmax, 1 for sample per character, 2 to sample per space
    import sys
    kwargs = {"model_ckpt": sys.argv[1],
              "prime": " ",
              "sample": 1,
              "sample_len": 80,
              "temperature": .35}
    sample(kwargs)
