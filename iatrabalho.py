import numpy as np
import tensorflow as tf
from midiutil import MIDIFile

# Parâmetros
batch_size = 1
seq_len = 128
n_notes = 128

# Definir o gerador (simplificado para o exemplo)
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(n_notes, activation='tanh')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

# Instanciar o gerador
generator = Generator()

# Gerar notas sintéticas
noise = tf.random.uniform([batch_size, seq_len], minval=0, maxval=1, dtype=tf.float32)
generated_notes = generator(noise)

# Converter as saídas do gerador para o intervalo 0-127 (notas MIDI)
generated_notes = (generated_notes + 1) * 63.5  # Transformação para o intervalo 0-127
generated_notes = tf.cast(generated_notes, tf.int32)  # Converter para inteiros

# Criar o arquivo MIDI
midi = MIDIFile(1)  # Um único canal
track = 0
time = 0  # Início da música

midi.addTrackName(track, time, "Track 1")
midi.addTempo(track, time, 120)  # BPM

# Adicionar notas ao arquivo MIDI
for i, note in enumerate(generated_notes.numpy().flatten()):
    midi.addNote(track, 0, int(note), time + i, 1, 100)  # Canal 0, duração 1, volume 100

# Escrever o arquivo MIDI
with open("generated_music.mid", "wb") as output_file:
    midi.writeFile(output_file)

print("Arquivo MIDI 'generated_music.mid' criado com sucesso!")
