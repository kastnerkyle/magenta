from __future__ import print_function
import tensorflow as tf
import os
from tfkdllib import numpy_softmax, numpy_sample_softmax, piano_roll_to_midi
from tfkdllib import write_out_midi_from_duration_pitch


def validate_sample_args(model_ckpt,
                         prime,
                         sample,
                         sample_len,
                         **kwargs):
    return (model_ckpt, prime, sample, sample_len)


def sample(kwargs):
    (model_ckpt,
     prime,
     sample,
     sample_len) = validate_sample_args(**kwargs)
    # Wow this is nastyyyyy
    from midi_rnn import *
    duration_mb, note_mb = train_itr.next()
    write_out_midi_from_duration_pitch("gt.mid", duration_mb, note_mb)

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
        i_h3 = np.zeros((batch_size, h_dim)).astype("float32")
        i_h2 = np.zeros((batch_size, h_dim)).astype("float32")
        i_h4 = np.zeros((batch_size, h_dim)).astype("float32")
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
        for j in range(len(note_inputs), sample_len):
            for ni in range(n_notes):
                feed = {note_inpt: note_inputs,
                        note_target: note_targets,
                        duration_inpt: duration_inputs,
                        duration_target: duration_targets,
                        init_h1: i_h1,
                        init_h2: i_h2,
                        init_h3: i_h3,
                        init_h4: i_h4}
                outs = []
                outs += note_preds
                outs += duration_preds
                outs += [final_h1, final_h2, final_h3, final_h4]
                r = sess.run(outs, feed)
                h_l = r[-4:]
                h1_l, h2_l, h3_l, h4_l = h_l
                this_preds = r[:-4]
                this_probs = [numpy_softmax(p) for p in this_preds]
                this_samples = [numpy_sample_softmax(p, random_state)
                                for p in this_probs]
                note_probs = this_probs[:n_notes]
                duration_probs = this_probs[n_notes:]
                # only put the single note in...
                full_notes[j, :, ni] = this_samples[ni].ravel()
                note_targets[0, :, ni] = this_samples[ni].ravel()
                # hacky hardcode
                full_durations[j, :, ni] = this_samples[ni + n_notes].ravel()
                duration_targets[0, :, ni] = this_samples[ni + n_notes].ravel()
            # priming sequence
            note_inputs = full_notes[j:j+1]
            duration_inputs = full_durations[j:j+1]
            note_targets = np.zeros((1, batch_size, train_itr.simultaneous_notes))
            duration_targets = np.zeros((1, batch_size, train_itr.simultaneous_notes))
            i_h1 = h1_l
            i_h2 = h2_l
            i_h3 = h3_l
            i_h4 = h4_l
        write_out_midi_from_duration_pitch("temp.mid", full_durations, full_notes)


if __name__ == '__main__':
    # prime is the text to prime with
    # sample is 0 for argmax, 1 for sample per character, 2 to sample per space
    import sys
    kwargs = {"model_ckpt": sys.argv[1],
              "prime": " ",
              "sample": 1,
              "sample_len": 1000}
    sample(kwargs)
