import wave
import time

from array import array
from collections.abc import Sequence
import meltysynth as ms


def write_wav_file(
    sample_rate: int, left: Sequence[float], right: Sequence[float], path: str
) -> None:
    max_value = 0.0

    for t in range(len(left)):
        if abs(left[t]) > max_value:
            max_value = abs(left[t])

        if abs(right[t]) > max_value:
            max_value = abs(right[t])

    a = 0.99 / max_value

    data = array("h")

    for t in range(len(left)):
        sample_left = int(32768 * a * left[t])
        sample_right = int(32768 * a * right[t])

        data.append(sample_left)
        data.append(sample_right)

    wav = wave.open(path, "wb")
    wav.setframerate(sample_rate)
    wav.setnchannels(2)
    wav.setsampwidth(2)
    wav.writeframesraw(data)
    wav.close()

def flourish() -> None:
    # Load the SoundFont.
    sound_font = ms.SoundFont.from_file("rockKit.sf2")

    # Create the synthesizer.
    settings = ms.SynthesizerSettings(44100)
    synthesizer = ms.Synthesizer(sound_font, settings)

    # Load the MIDI file.
    midi_file = ms.MidiFile.from_file("sambeat_orig.mid")

    # Create the MIDI sequencer.
    sequencer = ms.MidiFileSequencer(synthesizer)
    sequencer.play(midi_file, False)

    # The output buffer.
    left = ms.create_buffer(int(settings.sample_rate * midi_file.length))
    right = ms.create_buffer(int(settings.sample_rate * midi_file.length))

    # Render the waveform.
    start = time.time()
    sequencer.render(left, right)
    end = time.time()

    # Print the time elapsed.
    print("Time elapsed: " + str(end - start))

    # Save the waveform as a WAV file.
    write_wav_file(settings.sample_rate, left, right, "flourish.wav")


def main() -> None:
    flourish()


if __name__ == "__main__":
    main()