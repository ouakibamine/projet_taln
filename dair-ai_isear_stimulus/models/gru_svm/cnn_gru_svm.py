"""Implementation of the CNN + Bi-GRU + SVM model for Emotion Classification (TF 1.x compatibility)"""
from __future__ import absolute_import, division, print_function
import os
import warnings
from utils.data import plot_confusion_matrix
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


__version__ = "0.5.0"
__author__ = "Your Name"

import numpy as np
import os
import sys
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.nn.rnn_cell import GRUCell, DropoutWrapper
from tensorflow.keras import layers

class GruSvm:
    """CNN + BiGRU + SVM model for text emotion classification using TensorFlow 1.x API"""

    def __init__(
        self,
        alpha,
        batch_size,
        cell_size,
        dropout_rate,
        num_classes,
        sequence_length,
        svm_c,
        vocab_size,
        embedding_dim,
    ):
        self.alpha = alpha
        self.batch_size = batch_size
        self.cell_size = cell_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.svm_c = svm_c
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        def __graph__():
            # === Placeholders ===
            with tf.name_scope("input"):
                x_input = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_length], name="x_input")
                y_input = tf.placeholder(dtype=tf.uint8, shape=[None], name="y_input")
                y_onehot = tf.one_hot(indices=y_input, depth=self.num_classes, on_value=1.0, off_value=-1.0, name="y_onehot")

            # === Embedding layer ===
            with tf.name_scope("embedding"):
                embedding_matrix = tf.get_variable(
                    "embedding_matrix",
                    shape=[self.vocab_size, self.embedding_dim],
                    initializer=tf.random_uniform_initializer(-1.0, 1.0),
                )
                embedded_input = tf.nn.embedding_lookup(embedding_matrix, x_input)

            # === Initial state & learning rate ===
            p_keep = tf.placeholder(dtype=tf.float32, name="p_keep")
            learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

            # === CNN Branch ===
            with tf.name_scope("cnn_branch"):
                conv_layer = tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=5,
                activation=tf.nn.relu,
                padding='same'
            )
                conv = conv_layer(embedded_input)  # ✅ On applique la couche sur embedded_input
                pool = tf.reduce_max(conv, axis=1)
            
            # === GRU Branch ===
            with tf.name_scope("gru_branch"):
                # Create forward and backward cells
                cell_fw = GRUCell(self.cell_size)
                cell_bw = GRUCell(self.cell_size)
                
                # Apply dropout if needed
            if self.dropout_rate < 1.0:
                cell_fw = DropoutWrapper(cell_fw, input_keep_prob=self.dropout_rate, output_keep_prob=self.dropout_rate)
                cell_bw = DropoutWrapper(cell_bw, input_keep_prob=self.dropout_rate, output_keep_prob=self.dropout_rate)
                
                # Bidirectional RNN
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=embedded_input,
                    dtype=tf.float32
                )
                
                # Concatenate the final states from both directions
                last_fw = output_fw[:, -1, :]  # Forward final state
                last_bw = output_bw[:, -1, :]  # Backward final state
                gru_output = tf.concat([last_fw, last_bw], axis=1)
            
            # === Fusion des branches ===
            with tf.name_scope("merge"):
                merged = tf.concat([pool, gru_output], axis=1)
                #merged_size = pool.get_shape()[1] + gru_output.get_shape()[1]
                merged_size = int(pool.get_shape()[1]) + int(gru_output.get_shape()[1])

                # Couche dense intermédiaire optionnelle
                dense_layer = tf.keras.layers.Dense(
                    units=merged_size,
                    activation=tf.nn.relu
                )
                merged = dense_layer(merged)

            # === SVM classification layer ===
            with tf.name_scope("final_training_ops"):
                weight = tf.get_variable(
                    "weights",
                    shape=[merged_size, self.num_classes],  # ⚡ merged_size (CNN + GRU), pas juste cell_size
                    initializer=tf.random_normal_initializer(stddev=0.01),
                )
                bias = tf.get_variable(
                    "biases",
                    initializer=tf.constant(0.1, shape=[self.num_classes]),
                )
                logits = tf.matmul(merged, weight) + bias  # ✅ Utiliser merged ici
                tf.summary.histogram("pre_activations", logits)

            # === Loss: L2 + hinge ===
            with tf.name_scope("svm"):
                reg_loss = 0.5 * tf.reduce_sum(tf.square(weight))  # OK
                hinge = tf.maximum(0.0, 1.0 - y_onehot * logits)
                hinge_loss = tf.reduce_mean(tf.square(hinge))      # <=== changer sum -> mean
                loss = reg_loss + self.svm_c * hinge_loss
                tf.summary.scalar("loss", loss)
            # === Optimizer ===
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

            # === Accuracy ===
            with tf.name_scope("accuracy"):
                pred_sign = tf.sign(logits)
                pred_sign = tf.identity(pred_sign, name="prediction")
                correct = tf.equal(tf.argmax(pred_sign, 1), tf.argmax(y_onehot, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                tf.summary.scalar("accuracy", accuracy)

            # === Merge summaries ===
            merged = tf.summary.merge_all()

            # Expose tensors
            self.x_input = x_input
            self.y_input = y_input
            self.y_onehot = y_onehot
            self.p_keep = p_keep
            self.learning_rate = learning_rate
            self.logits = logits
            self.loss = loss
            self.optimizer = optimizer
            self.prediction = pred_sign
            self.accuracy = accuracy
            self.merged = merged

        sys.stdout.write("\n<log> Building Graph...")
        __graph__()
        sys.stdout.write("</log>\n")

   

    def train(self, checkpoint_path, log_path, model_name, epochs, train_data, train_size, validation_data, validation_size, result_path):
        print("Début de l'entraînement...")

        # Créer les répertoires nécessaires s'ils n'existent pas
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        saver = tf.train.Saver(max_to_keep=5)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        train_writer = tf.summary.FileWriter(os.path.join(log_path, "train"), graph=tf.get_default_graph())
        val_writer = tf.summary.FileWriter(os.path.join(log_path, "val"), graph=tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(init_op)

            # Restaurer un checkpoint existant si disponible
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
                print("Checkpoint restauré.")

            

            # Entraînement
            for step in range(epochs * train_size // self.batch_size):
                offset = (step * self.batch_size) % train_size
                bx = train_data[0][offset: offset + self.batch_size]
                by = train_data[1][offset: offset + self.batch_size]

                feed = {
                    self.x_input: bx,
                    self.y_input: by,
                    self.learning_rate: self.alpha,
                    self.p_keep: self.dropout_rate,  # Dropout pour l'entraînement
                }

                # Exécuter une itération d'entraînement
                summary, _, preds, acts = sess.run(
                    [self.merged, self.optimizer, self.prediction, self.y_onehot],
                    feed_dict=feed,
                )

                # Affichage et sauvegarde périodique
                # Affichage et sauvegarde périodique
                if step % 100 == 0:
                    # Calcul sur le batch d'entraînement
                    l_train, a_train = sess.run([self.loss, self.accuracy], feed_dict=feed)

                    # Calcul sur tout le set de validation
                    val_feed = {
                        self.x_input: validation_data[0],
                        self.y_input: validation_data[1],
                        self.learning_rate: self.alpha,
                        self.p_keep: 1.0,  # Pas de dropout en validation
                    }
                    l_val, a_val = sess.run([self.loss, self.accuracy], feed_dict=val_feed)
                    
                    print(f"step [{step}] | train loss={l_train:.4f} train acc={a_train:.4f} | val loss={l_val:.4f} val acc={a_val:.4f}")

                    # Ajouter les résumés
                    train_writer.add_summary(summary, step)
                    saver.save(sess, os.path.join(checkpoint_path, model_name), global_step=step)


                # Sauvegarder les prédictions et les vraies étiquettes
                self.save_labels(preds, acts, result_path, step, "training")

            print("Entraînement terminé.")
            # Validation
            total_val_loss = 0
            total_val_acc = 0
            num_batches = validation_size // self.batch_size
            all_preds = []  # Pour stocker toutes les prédictions
            all_acts = [] 

            for step in range(num_batches):
                offset = (step * self.batch_size) % validation_size
                bx = validation_data[0][offset: offset + self.batch_size]
                by = validation_data[1][offset: offset + self.batch_size]

                feed = {
                    self.x_input: bx,
                    self.y_input: by,
                    self.p_keep: 1.0,  # Pas de dropout pendant la validation
                }

                summary, preds, acts, l, a = sess.run(
                    [self.merged, self.prediction, self.y_onehot, self.loss, self.accuracy],
                    feed_dict=feed,
                )

                total_val_loss += l
                total_val_acc += a
                # Stocker les prédictions et les vraies étiquettes pour la matrice de confusion
                all_preds.append(preds)
                all_acts.append(acts)

                if step % 100 == 0:
                    print(f"step [{step}] val loss={l:.4f} val acc={a:.4f}")
                    val_writer.add_summary(summary, step)

                self.save_labels(preds, acts, result_path, step, "validation")

            # Fin de validation : afficher les moyennes
            avg_val_loss = total_val_loss / num_batches
            avg_val_acc = total_val_acc / num_batches
            print(f"Validation: avg loss={avg_val_loss:.4f}, avg acc={avg_val_acc:.4f}")

            # Convertir les listes de prédictions et d'actes en arrays numpy
            all_preds = np.concatenate(all_preds, axis=0)
            all_acts = np.concatenate(all_acts, axis=0)

            # Définir le chemin où sauvegarder
            save_path = os.path.join(result_path, "preds_acts.txt")

            # Ouvrir le fichier et écrire les prédictions et actes
            with open(save_path, "w") as f:
                f.write("Prediction\tActual\n")  # en-tête
                for pred, act in zip(all_preds, all_acts):
                    f.write(f"{pred}\t{act}\n")

            print("Validation terminée.")


    @staticmethod
    def predict(batch_size, cell_size, dropout_rate, num_classes, test_data, test_size, checkpoint_path, result_path):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                sv = tf.train.Saver()
                sv.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
                print(f"Loaded model {tf.train.latest_checkpoint(checkpoint_path)}")

            for step in range(test_size // batch_size):
                offset = (step * batch_size) % test_size
                bx = test_data[0][offset: offset + batch_size]
                by = test_data[1][offset: offset + batch_size]
                yoh = sess.run(tf.one_hot(by, num_classes, 1.0, -1.0))

                feed = {
                    "input/x_input:0": bx,
                    "p_keep:0": dropout_rate,
                }
                pt = sess.graph.get_tensor_by_name("accuracy/prediction:0")
                preds = sess.run(pt, feed_dict=feed)
                at = sess.graph.get_tensor_by_name("accuracy/accuracy/Mean:0")
                acc = sess.run(at, feed_dict={**feed, "input/y_input:0": by})

                if step % 100 == 0:
                    print(f"step [{step}] test acc={acc:.4f}")
                GruSvm.save_labels(preds, yoh, result_path, step, "testing")

    @staticmethod
    def save_labels(predictions, actual, result_path, step, phase):
        labels = np.concatenate((predictions, actual), axis=1)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        np.save(os.path.join(result_path, f"{phase}-gru_svm-{step}.npy"), labels)
