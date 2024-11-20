import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources if not already installed
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

nltk.data.path.append("C:\\Users\\Ashkan Ansarifard\\AppData\\Roaming\\nltk_data")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


class TextPreprocessor:
    def __init__(self, enable_multi_word_terms=True, enable_stopwords=True, enable_punctuation_removal=True,
                 enable_joined_terms=True, enable_processing=True, enable_restoration=True):
        # Enable or disable specific processing steps
        self.enable_multi_word_terms = enable_multi_word_terms
        self.enable_stopwords = enable_stopwords
        self.enable_punctuation_removal = enable_punctuation_removal
        self.enable_joined_terms = enable_joined_terms
        self.enable_processing = enable_processing
        self.enable_restoration = enable_restoration

        # Initialize English stopwords and add some common Italian stopwords
        self.stop_words = set(stopwords.words("english")).union({
            'di', 'da', 'con', 'fino', 'generazion', 'pollici', 'per', 'su', 'il', 'la', 'le', 'un', 'uno', 'una',
            'che', 'e', 'ma', 'non', 'si', 'ci', 'lo', "c'è", 'cm'
        })
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.unit_keywords = {'gb', 'tb', 'mb', 'ghz', 'ddr', 'ram', 'ssd', 'hz', 'mhz'}  # Common units

        # Custom list of words to retain in their original form
        self.preserved_words = {'laptop', 'portatile', 'office', 'windows', 'intel', 'amd', 'mouse', 'computer',
                                'home', 'pc', 'personal computer', 'apple', 'mac'}

        # Multi-word terms to preserve as single tokens
        self.multi_word_terms = [
            # Operating Systems
            "windows 11 pro", "windows 11 home", "windows 10 pro", "windows 10 home",
            "win 11 pro", "win 11 home", "win 10 pro", "win 10 home",
            "win 11", "win 10",
            "chrome os", "mac os", "linux ubuntu",

            # Office Suites
            "office 365", "office 2019", "office 19 pro", "office 2021", "office 2021 pro", "libre office",
            "google workspace",
            "365 office", "office 19", "office 21", "office 19 pro", "office 21 pro", "office 2021 pro"

            # Processor Models
                                                                                      "intel core i3", "core i3",
            "intel core i5", "core i5", "intel core i7", "core i7", "intel core i9", "core i9",
            "amd ryzen 3", "amd ryzen 5", "amd ryzen 7", "amd ryzen 9",
            "apple m1", "apple m2",

            # Graphics Cards
            "nvidia geforce gtx", "nvidia geforce rtx", "amd radeon rx",
            "intel iris xe", "nvidia quadro", "integrated graphics", "discrete graphics",

            # Display Terms
            "full hd", "4k uhd", "8k uhd", "oled display", "retina display",
            "anti glare", "infinity edge", "touch screen", "refresh rate",

            # Storage and Memory
            "ddr4 ram", "ddr5 ram", "lpddr4x", "lpddr5x", "m.2 ssd", "nvme ssd",
            "sata ssd", "hard drive", "hdd", "1tb ssd", "2tb ssd", "512gb ssd", "256gb ssd",
            "16gb ram", "32gb ram", "64gb ram", "128gb ssd", "64gb emmc", "expandable storage",

            # Networking and Connectivity
            "dual band wifi", "wifi 6", "wifi 5", "bluetooth 5.1", "bluetooth 5.2",
            "ethernet port", "gigabit ethernet", "usb c", "usb 3.0", "hdmi port",
            "thunderbolt 3", "thunderbolt 4", "wi-fi"

            # Battery and Power
                                              "battery life", "fast charging", "usb c charging", "power adapter",
            "watt power supply",
            "battery backup", "removable battery", "long battery life",

            # Dimensions and Build
            "lightweight design", "slim bezel", "aluminum body", "carbon fiber",
            "spill resistant keyboard", "backlit keyboard", "fingerprint reader",
            "face recognition", "secure boot", "fanless design", "portable design",

            # Audio and Multimedia
            "dolby audio", "stereo speakers", "dual speakers", "noise cancellation",
            "built in webcam", "hd webcam", "privacy shutter", "dual microphone",

            # Accessories and Extras
            "wireless mouse", "keyboard cover", "stylus pen", "usb hub", "hdmi cable",
            "screen protector", "carry case", "laptop stand",

            # Miscellaneous Phrases
            "gaming laptop", "workstation laptop", "business laptop", "2 in 1 laptop",
            "detachable keyboard", "convertible laptop", "all in one desktop", "tower pc",
            "mini pc", "high performance", "energy efficient", "overclocked", "pre installed software"

            # PC Types
                                                                              "mini pc", "mini computer",
            "mini desktop", "pc portatile", "all in one", "pronto all'uso"
        ]

    def preprocess_multi_word_terms(self, text):
        """
        Temporarily replace spaces in multi-word terms with underscores to preserve them as single tokens.
        """
        if not self.enable_multi_word_terms:
            return text
        for phrase in self.multi_word_terms:
            modified_phrase = phrase.replace(" ", "_")
            text = re.sub(r'\b' + re.escape(phrase) + r'\b', modified_phrase, text, flags=re.IGNORECASE)
        return text

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_punctuation_and_symbols(self, tokens):
        """
        Removes punctuation and symbols from tokens, retains alphanumeric units, and separates numbers from letters.
        """
        if not self.enable_punctuation_removal:
            return tokens
        cleaned_tokens = []
        for token in tokens:
            cleaned_token = re.sub(r"[^\w\s]", "", token)
            if cleaned_token.isdigit() or cleaned_token.lower() in self.unit_keywords:
                continue
            if re.match(r"\d+[a-zA-Z]+", cleaned_token):
                cleaned_tokens.append(cleaned_token.lower())
            elif cleaned_token:
                cleaned_tokens.append(cleaned_token.lower())
        return cleaned_tokens

    def handle_joined_terms(self, tokens):
        """
        Join terms like 'm2' and 'ssd' into 'm.2 ssd' or 'dual band' into one phrase.
        """
        if not self.enable_joined_terms:
            return tokens
        joined_tokens = []
        skip_next = False
        for i in range(len(tokens) - 1):
            if skip_next:
                skip_next = False
                continue
            current_token = tokens[i]
            next_token = tokens[i + 1]
            if current_token == "m2" and next_token == "ssd":
                joined_tokens.append("m.2 ssd")
                skip_next = True
            elif current_token == "dual" and next_token == "band":
                joined_tokens.append("dual-band")
                skip_next = True
            else:
                joined_tokens.append(current_token)
        if not skip_next:
            joined_tokens.append(tokens[-1])
        return joined_tokens

    def remove_stopwords(self, tokens):
        if not self.enable_stopwords:
            return tokens
        return [token for token in tokens if token.lower() not in self.stop_words]

    def process_tokens(self, tokens):
        """
        Process tokens using lemmatization for preserved words and stemming for others.
        :param tokens: list of str, input tokens
        :return: list of str, processed tokens
        """
        if not self.enable_processing:
            return tokens
        processed_tokens = []
        for token in tokens:
            if token.lower() in self.preserved_words:
                processed_tokens.append(self.lemmatizer.lemmatize(token))
            else:
                processed_tokens.append(self.stemmer.stem(token))
        return processed_tokens

    def restore_multi_word_terms(self, tokens):
        """
        Restore underscores to spaces for multi-word terms that were preserved.
        """
        if not self.enable_processing:
            return tokens
        processed_tokens = []
        for token in tokens:
            if token.lower() in self.preserved_words:
                processed_tokens.append(self.lemmatizer.lemmatize(token))
            else:
                processed_tokens.append(self.stemmer.stem(token))
        return processed_tokens

    def preprocess_text(self, text):
        text = self.preprocess_multi_word_terms(text) if self.enable_multi_word_terms else text
        tokens = self.tokenize(text)
        tokens = self.remove_punctuation_and_symbols(tokens) if self.enable_punctuation_removal else tokens
        tokens = self.handle_joined_terms(tokens) if self.enable_joined_terms else tokens
        tokens = self.remove_stopwords(tokens) if self.enable_stopwords else tokens
        tokens = self.process_tokens(tokens) if self.enable_processing else tokens
        tokens = self.restore_multi_word_terms(tokens) if self.enable_restoration else tokens
        return tokens
