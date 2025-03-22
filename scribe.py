import speech_recognition as sr
import transformers
import torch

class AIScribe:
    def __init__(self, summary_model_name="facebook/bart-large-cnn", translation_model_name="Helsinki-NLP/opus-mt-en-fr"):
        """
        Initializes the AIScribe with speech recognition, summarization, and translation capabilities.

        Args:
            summary_model_name (str): Name of the summarization model from Hugging Face Transformers.
            translation_model_name (str): Name of the translation model from Hugging Face Transformers.
        """
        self.recognizer = sr.Recognizer()
        self.summary_model = transformers.pipeline("summarization", model=summary_model_name)
        self.translation_model = transformers.pipeline("translation", model=translation_model_name)

    def transcribe_audio(self, audio_file_path):
        """
        Transcribes audio from a file.

        Args:
            audio_file_path (str): Path to the audio file.

        Returns:
            str: Transcribed text, or None if an error occurs.
        """
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                return text
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None
        except FileNotFoundError:
            print(f"File not found: {audio_file_path}")
            return None

    def summarize_text(self, text):
        """
        Summarizes the given text.

        Args:
            text (str): Text to summarize.

        Returns:
            str: Summarized text.
        """
        summary = self.summary_model(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']

    def translate_text(self, text, target_language="fr"):
        """
        Translates the given text to the target language.

        Args:
            text (str): Text to translate.
            target_language (str): Target language code (e.g., 'fr' for French, 'es' for Spanish).

        Returns:
            str: Translated text.
        """
        if target_language == "fr":
            translated = self.translation_model(text)[0]['translation_text']
            return translated
        else:
            print("Target language not currently supported. Only French is supported")
            return "Target language not supported"

    def process(self, input_data, task="transcribe", target_language="fr"):

        """
        Processes the input data based on the specified task.

        Args:
            input_data (str): Path to audio file or text.
            task (str): Task to perform ('transcribe', 'summarize', 'translate').
            target_language (str) : target language for translation.
        Returns:
            str: Processed output.
        """
        if task == "transcribe":
            return self.transcribe_audio(input_data)
        elif task == "summarize":
            return self.summarize_text(input_data)
        elif task == "translate":
            return self.translate_text(input_data, target_language)
        else:
            return "Invalid task."

# Example Usage
if __name__ == "__main__":
    scribe = AIScribe()

    #Example audio transcription. Place your own audio file.
    audio_file = "example.wav" #replace with your file
    transcribed_text = scribe.process(audio_file, task="transcribe")
    if transcribed_text:
        print("Transcribed Text:", transcribed_text)

    # Example text summarization.
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text that needs to be summarized. It has a lot of information that is not very useful. 
    We want to condense this text into a shorter, more concise version that captures the main points.
    """
    summary = scribe.process(sample_text, task="summarize")
    print("\nSummary:", summary)

    #example translation
    translation = scribe.process("Hello, how are you?", task = "translate")
    print("\nTranslation:", translation)