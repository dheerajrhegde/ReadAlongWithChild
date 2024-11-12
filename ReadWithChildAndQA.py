import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import pygame
import pyttsx3
import speech_recognition as sr
import threading
import difflib
from gtts import gTTS
import os
import openai
import textwrap, json
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

class TextExtractor:
    """
    A class to extract text from images using pytesseract.
    """
    def __init__(self):
        """
        Initializes the TextExtractor class and sets the path for Tesseract.
        """
        pytesseract.pytesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.4.1_1/bin/tesseract'  # Update the path if necessary

    def extract_words_coordinates(self, image):
        """
        Extracts words and their coordinates from an image.

        Args:
            image (PIL.Image): The image from which to extract words.

        Returns:
            list: A list of dictionaries containing word and its coordinates (x, y, w, h).
        """
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                word = data['text'][i]
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                words.append({'word': word, 'x': x, 'y': y, 'w': w, 'h': h})
        return words

    def extract_sentences_coordinates(self, image):
        """
        Extracts sentences and their coordinates from an image.

        Args:
            image (PIL.Image): The image from which to extract sentences.

        Returns:
            list: A list of dictionaries containing sentence and its coordinates.
        """
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        sentences = []
        current_sentence = ""
        current_coords = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                word = data['text'][i]
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                current_sentence += word + " "
                current_coords.append({'x': x, 'y': y, 'w': w, 'h': h})
                if word.endswith(('.', '!', '?')):
                    sentences.append({'sentence': current_sentence.strip(), 'coords': current_coords})
                    current_sentence = ""
                    current_coords = []
        if current_sentence:
            sentences.append({'sentence': current_sentence.strip(), 'coords': current_coords})
        return sentences

class SpeechRecognizer:
    """
    A class to handle speech recognition and similarity checking.
    """
    def recognize_speech(self, recognizer, source, result_dict):
        """
        Handles speech recognition in a separate thread.

        Args:
            recognizer (speech_recognition.Recognizer): The recognizer instance.
            source (speech_recognition.AudioSource): The audio source to listen to.
            result_dict (dict): A dictionary to store the result of the recognition.
        """
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)  # Reduced timeout and added phrase time limit
            result_dict['spoken_text'] = recognizer.recognize_google(audio).strip().lower()
        except Exception as e:
            result_dict['error'] = str(e)

    def is_text_similar(self, spoken_text, target_text, threshold=0.8):
        """
        Checks the similarity between spoken text and target text.

        Args:
            spoken_text (str): The text spoken by the user.
            target_text (str): The target text to compare against.
            threshold (float): The similarity threshold (default is 0.8).

        Returns:
            bool: True if the similarity is above the threshold, False otherwise.
        """
        similarity = difflib.SequenceMatcher(None, spoken_text, target_text).ratio()
        return similarity >= threshold

class PDFHighlighter:
    """
    A class to handle PDF highlighting and speech interaction.
    """
    def __init__(self, grade_level=1):
        """
        Initializes the PDFHighlighter class.

        Args:
            grade_level (int): The grade level for generating questions (default is 1).
        """
        self.ie = TextExtractor()
        self.speech = SpeechRecognizer()
        self.grade_level = grade_level
        self.text_content = ""
        self.correct_answers = 0
        self.total_questions = 0

    def log(self, text):
        """
        Logs the text that was not spoken correctly.

        Args:
            text (str): The text that was not spoken correctly.
        """
        print("This sentence was not spoken correctly: ", text)

    def select_level(self):
        """
        Creates a UI to select the reading level.

        Returns:
            int: The selected reading level (1 for word-by-word, 2 for sentence-by-sentence), or None if exited.
        """
        pygame.init()
        screen = pygame.display.set_mode((600, 400))
        pygame.display.set_caption('Select Reading Level')

        font = pygame.font.Font(None, 24)
        level_1_text = font.render('Level 1 (Word-by-Word)', True, (0, 0, 0))
        level_2_text = font.render('Level 2 (Sentence-by-Sentence)', True, (0, 0, 0))

        level_1_button = pygame.Rect(100, 100, 400, 80)
        level_2_button = pygame.Rect(100, 220, 400, 80)

        running = True
        selected_level = None

        while running:
            screen.fill((255, 255, 255))

            # Draw buttons
            pygame.draw.rect(screen, (200, 200, 200), level_1_button)
            pygame.draw.rect(screen, (200, 200, 200), level_2_button)

            # Render text onto buttons
            screen.blit(level_1_text, (110, 110))
            screen.blit(level_2_text, (110, 230))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if level_1_button.collidepoint(event.pos):
                        selected_level = 1
                        running = False
                    elif level_2_button.collidepoint(event.pos):
                        selected_level = 2
                        running = False

        pygame.quit()
        return selected_level

    def display_and_read(self, input_pdf_path):
        """
        Loads the PDF and displays each page with highlighted words or sentences using Pygame.

        Args:
            input_pdf_path (str): The path to the PDF file to be displayed and read.
        """
        level = self.select_level()
        if level is None:
            print("User exited the level selection.")
            return

        # Initialize Pygame, TTS engine, and Speech Recognition
        pygame.init()
        engine = pyttsx3.init()
        recognizer = sr.Recognizer()
        engine.setProperty('rate', 200)  # Increase the speech rate to speed up the process
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'english' in voice.languages and 'us' in voice.id:
                engine.setProperty('voice', voice.id)
                break

        # Open the PDF
        pdf_document = fitz.open(input_pdf_path)

        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)

            # Render the page to an image for Pygame display
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            mode = img.mode
            size = img.size
            data = img.tobytes()
            pygame_image = pygame.image.fromstring(data, size, mode)

            # Set up Pygame window
            screen = pygame.display.set_mode(size)
            pygame.display.set_caption(f'Page {page_number + 1}')

            if level == 1:
                # Extract words and their coordinates
                elements = self.ie.extract_words_coordinates(img)
                element_key = 'word'
            else:
                # Extract sentences and their coordinates
                elements = self.ie.extract_sentences_coordinates(img)
                element_key = 'sentence'

            running = True
            element_index = 0
            attempts = 0

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        pygame.quit()
                        return

                # Display the PDF page
                screen.blit(pygame_image, (0, 0))

                # Highlight the current word or sentence
                if element_index < len(elements):
                    element_info = elements[element_index]
                    if level == 1:
                        coords = [{'x': element_info['x'], 'y': element_info['y'], 'w': element_info['w'], 'h': element_info['h']}]
                    else:
                        coords = element_info['coords']

                    for coord in coords:
                        x, y, w, h = coord['x'], coord['y'], coord['w'], coord['h']
                        highlight_rect = pygame.Rect(x - 2, y - 2, w + 4, h + 4)
                        highlight_surface = pygame.Surface((highlight_rect.width, highlight_rect.height))
                        highlight_surface.set_alpha(100)  # Transparency level
                        highlight_surface.fill((255, 255, 0))  # Yellow color
                        screen.blit(highlight_surface, highlight_rect.topleft)

                    # Update the display before listening for the word or sentence
                    pygame.display.flip()

                    # Listen for the word or sentence in a separate thread to speed up recognition
                    with sr.Microphone() as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.3)  # Reduce ambient noise impact duration
                        result_dict = {}
                        recognition_thread = threading.Thread(target=self.speech.recognize_speech, args=(recognizer, source, result_dict))
                        recognition_thread.start()
                        recognition_thread.join()

                        if 'spoken_text' in result_dict:
                            spoken_text = result_dict['spoken_text']
                            print(spoken_text)
                            actual_text = element_info[element_key].strip().strip('.,!?;:"').lower()

                            if self.speech.is_text_similar(spoken_text, actual_text):
                                element_index += 1
                                attempts = 0  # Reset attempts for the next element
                            else:
                                # If incorrect, use gTTS to say the word or sentence aloud
                                tts = gTTS(text=actual_text, lang='en')
                                tts.save("temp.mp3")
                                os.system("mpg321 temp.mp3")
                                os.remove("temp.mp3")
                                attempts += 1
                                if attempts >= 2:
                                    element_index += 1
                                    attempts = 0
                                self.log(actual_text)
                                self.text_content += actual_text + " "
                        elif 'error' in result_dict:
                            print(result_dict['error'])
                            actual_text = element_info[element_key].strip().strip('.,!?;:"')
                            tts = gTTS(text=actual_text, lang='en')
                            tts.save("temp.mp3")
                            os.system("mpg321 temp.mp3")
                            os.remove("temp.mp3")
                            element_index += 1
                            attempts = 0
                            self.log(actual_text)
                            self.text_content += actual_text + " "
                else:
                    running = False

        # After reading all pages, ask questions
        self.ask_questions()

    def ask_questions(self):
        """
        Generate questions based on the text content after reading the entire PDF using OpenAI API.
        """
        print("in ask_questions")
        if not self.text_content.strip():
            print("No content available for generating questions.")
            return

        parser = JsonOutputParser()

        question_prompt = PromptTemplate(
            template="""Generate 3 questions and its answer suitable for grade {grade} based on the following story: {content}.
                      Output should be in {format_instructions}"
                      Ask questions without any personal pronouns."
                      Ask questions that relate to key part of the story. 
                      Don't use complex words that are not in the story. """,
            input_variables=["grade", "content"],
            partial_variables={"format_instructions": parser.get_format_instructions()},)


        model = ChatOpenAI(temperature=0)
        chain = question_prompt | model | parser

        data = chain.invoke({"grade": self.grade_level, "content":self.text_content })["questions"]

        """print("response from chain is -> ",type(response),response)

        data = json.loads(response["questions"])"""

        pygame.init()
        screen = pygame.display.set_mode((600, 400))
        pygame.display.set_caption('Questions')

        font = pygame.font.Font(None, 24)
        recognizer = sr.Recognizer()
        for qna in data:
            question = qna["question"]

            wrapped_question = textwrap.fill(question, width=50)
            screen.fill((255, 255, 255))
            y_offset = 150
            for line in wrapped_question.split('\n'):
                question_text = font.render(line, True, (0, 0, 0))
                screen.blit(question_text, (50, y_offset))
                y_offset += 40
            pygame.display.flip()

            # Listen for child's response
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                result_dict = {}
                recognition_thread = threading.Thread(target=self.speech.recognize_speech, args=(recognizer, source, result_dict))
                recognition_thread.start()
                recognition_thread.join()

                if 'spoken_text' in result_dict:
                    child_response = result_dict['spoken_text']
                    print(f"Child's response: {child_response}")

                    # Use OpenAI to check if the response is correct
                    check_prompt = f"Based on the following story text, is the child's response correct?\n\nStory Text: {self.text_content}\n\nQuestion: {question}\n\nChild's Response: {child_response}\n\nAnswer with 'Yes' or 'No' and provide a brief explanation."
                    check_result = model.invoke(check_prompt)#check_response.choices[0].text.strip()
                    print("check_result ",check_result.content)
                    print(type(check_result))
                    if check_result.content.lower().startswith('yes'):
                        self.correct_answers += 1
                elif 'error' in result_dict:
                    print(f"Error recognizing speech: {result_dict['error']}")

                self.total_questions += 1

        # Display the score
        self.display_score()

    def display_score(self):
        """
        Display the score on the screen after all questions are answered.
        """
        pygame.init()
        screen = pygame.display.set_mode((600, 400))
        pygame.display.set_caption('Score')

        font = pygame.font.Font(None, 48)
        score_text = font.render(f'Score: {self.correct_answers} / {self.total_questions}', True, (0, 0, 0))
        close_button = pygame.Rect(200, 300, 200, 50)
        close_text = font.render('Close', True, (255, 255, 255))

        running = True
        while running:
            screen.fill((255, 255, 255))
            screen.blit(score_text, (150, 100))
            pygame.draw.rect(screen, (0, 0, 0), close_button)
            screen.blit(close_text, (close_button.x + 50, close_button.y + 10))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if close_button.collidepoint(event.pos):
                        running = False

        pygame.quit()

# Example usage
if __name__ == "__main__":
    story = PDFHighlighter(grade_level=1)
    story.display_and_read("duck 1 pager.pdf")
