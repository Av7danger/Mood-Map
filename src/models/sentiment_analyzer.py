import joblib
import torch
from transformers import pipeline, RobertaForSequenceClassification, BartForSequenceClassification
import numpy as np
import re
import datetime
import os
import json
from src.utils.logging_utils import setup_logging
from src.utils.rag_enhancer import initialize_sentiment_rag

# Setup logging
logger = setup_logging("logs/model_logs.log")

class SentimentModel:
    """A wrapper around Hugging Face's sentiment analysis pipeline for nuanced sentiment analysis."""
    
    def __init__(self, model_type="roberta", safe=False, use_rag=True, use_bart=False):
        """Initialize the sentiment analyzer using a pre-trained model.
        
        Args:
            model_type: The model type to use ('roberta' or 'bart')
            safe: If True, don't initialize the analyzer pipeline (useful for inheritance)
            use_rag: If True, use Retrieval Augmented Generation for enhanced predictions
            use_bart: If True, use BART for text summarization as part of the pipeline
        """
        self.model_type = model_type
        self.use_rag = use_rag
        self.use_bart = use_bart
        
        # Log model configuration
        logger.info(f"Initializing SentimentModel with: model_type={model_type}, use_rag={use_rag}, use_bart={use_bart}")
        
        # Initialize RAG if requested
        if use_rag:
            try:
                self.rag = initialize_sentiment_rag()
                logger.info("RAG initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing RAG: {str(e)}")
                self.rag = None
        else:
            self.rag = None
            
        # Initialize BART if requested
        if use_bart:
            try:
                from transformers import BartForConditionalGeneration, BartTokenizer
                self.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
                self.bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
                self.bart_pipeline = pipeline("summarization", model=self.bart_model, tokenizer=self.bart_tokenizer)
                logger.info("BART initialized successfully with RAG")
            except Exception as e:
                logger.error(f"Error initializing BART: {str(e)}")
                self.use_bart = False
        
        # Use a model specifically fine-tuned for sentiment analysis
        if model_type == "roberta":
            self.model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
            logger.info("Loaded RoBERTa sentiment model with RAG")
        elif model_type == "bart":
            # Only use BART for sentiment if not already using it for summarization
            self.model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
            logger.info("Loaded BART sentiment model with RAG")
        else:
            raise ValueError("Invalid model type. Choose 'roberta' or 'bart'.")
        
        # Try to use CUDA if available
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Device set to use {'CUDA' if self.device >= 0 else 'CPU'}")
        
        # Create the sentiment analysis pipeline (unless in safe mode)
        if not safe:
            self.analyzer = pipeline("sentiment-analysis", model=self.model, device=self.device)
            logger.info("Sentiment analysis pipeline created")
        else:
            # In safe mode, don't initialize analyzer
            self.analyzer = None
            logger.info("Initialized in safe mode - analyzer pipeline not created")
        
        # Define the sentiment categories - UPDATED TO 3 CLASSES FROM 5
        self.sentiment_categories = [
            "negative",
            "neutral",
            "positive"
        ]
        
        # Enhanced and expanded keyword lists for better sentiment detection
        self.neutral_keywords = [
            "okay", "ok", "fine", "alright", "average", "mediocre", 
            "ordinary", "decent", "satisfactory", "acceptable", "fair",
            "moderate", "so-so", "nothing special", "standard", "usual",
            "typical", "common", "regular", "middle", "meh", "neither", "mixed",
            "bland", "indifferent", "passable", "unremarkable", "tolerable",
            "sufficient", "adequate", "not bad", "functional", "conventional",
            "plain", "conventional", "neutral", "so so", "whatever", "just ok",
            "forgettable", "unexceptional", "flat", "tepid", "lukewarm"
        ]
        
        self.positive_keywords = [
            "good", "nice", "enjoyed", "pleasant", "recommend", "happy", "satisfied",
            "helpful", "great", "positive", "like", "useful", "comfortable", 
            "convenient", "effective", "appealing", "worthwhile", "solid", 
            "reliable", "smooth", "commendable", "appreciate", "pleasing",
            "gratifying", "favorable", "agreeable", "likable", "worthy",
            "satisfying", "well-made", "well made", "beneficial", "worthy",
            "improved", "enjoyable", "deserves", "pleased", "pretty good",
            "value", "valuable", "handy", "practical", "efficient", "well done",
            # Adding new social media positive terms
            "vibe", "vibey", "W", "dub", "valid", "iconic", "slay", "flex", 
            "wholesome", "based", "goated", "bussin", "chef's kiss", "elite",
            "hits different", "clean", "fresh", "quality", "superior", "100",
            "stan", "underrated", "top tier", "bet", "facts", "legit"
        ]
        
        self.very_positive_keywords = [
            "amazing", "excellent", "outstanding", "fantastic", "wonderful", "brilliant",
            "exceptional", "incredible", "superb", "phenomenal", "perfect", "thrilled",
            "best", "love", "awesome", "delighted", "exceptional", "marvelous",
            "spectacular", "terrific", "magnificent", "sublime", "exquisite",
            "impressive", "remarkable", "splendid", "stellar", "top-notch",
            "extraordinary", "breathtaking", "astounding", "stunning", "flawless",
            "peerless", "unmatched", "unsurpassed", "mind-blowing", "life-changing",
            "unparalleled", "revolutionary", "transcendent", "impeccable", "game-changer",
            "ecstatic", "overjoyed", "elated", "beyond expectations", "exceeded expectations",
            "adore", "fantastic", "fabulous", "astonishing", "blown away", "awe-inspiring",
            "masterful", "masterpiece", "couldn't be better", "absolutely loved",
            # Adding new social media very positive terms
            "GOAT", "fire", "lit", "god tier", "absolutely goated", "insane",
            "immaculate", "unreal", "elite", "next level", "top tier", "god tier",
            "legendary", "cracked", "immaculate vibes", "chef's kiss", "no notes",
            "hitting", "slaps", "straight fire", "undefeated", "peak", "goes hard",
            "banger", "absolute W", "massive W", "certified fresh"
        ]
        
        self.negative_keywords = [
            "bad", "poor", "disappointing", "disappointed", "issues", "problems",
            "unpleasant", "unhappy", "dissatisfied", "disliked", "let down",
            "mediocre", "subpar", "insufficient", "frustrating", "flawed",
            "annoying", "irritating", "lacking", "deficient", "underwhelming",
            "unsatisfying", "unfortunate", "inferior", "not worth", "overpriced",
            "inconvenient", "lackluster", "ineffective", "inefficient",
            "regret", "regrettable", "shortcomings", "defects", "disappoints",
            "fails", "failed", "frustrates", "frustration", "bothersome",
            "troubled", "concerning", "worse", "overrated", "below average",
            "not good", "not great", "not recommended", "not impressed",
            # Adding new social media negative terms
            "meh", "mid", "not it", "ain't it", "basic", "weak", "lame", "yikes",
            "yikes", "sketchy", "sus", "shady", "questionable", "overhyped",
            "cap", "capping", "wack", "clapped", "L", "taking an L", "ratio",
            "fell off", "average", "lowkey bad", "not the move", "down bad",
            "touch grass", "nah", "pass", "skipping", "unimpressed", "not worth"
        ]
        
        self.very_negative_keywords = [
            "terrible", "horrible", "awful", "worst", "hate", "disgusting", "avoid",
            "dreadful", "appalling", "atrocious", "abysmal", "garbage", "waste",
            "useless", "rubbish", "never", "pathetic", "disastrous", "catastrophic",
            "deplorable", "despicable", "inexcusable", "unbearable", "unacceptable",
            "intolerable", "miserable", "horrific", "laughable", "ridiculous",
            "worthless", "revolting", "repulsive", "repugnant", "offensive",
            "shoddy", "sucks", "loathe", "detestable", "nightmarish",
            "traumatic", "insufferable", "infuriating", "outrageous", "shameful",
            "failure", "mess", "total disaster", "complete failure", "absolutely terrible",
            "horrifying", "hideous", "catastrophe", "the worst ever", "absolutely hate",
            "garbage", "trash", "junk", "worst experience", "stay away", "run away from",
            "never again", "beyond awful", "completely useless", "utter disappointment",
            # Adding new social media very negative terms
            "trash", "pure garbage", "dogwater", "dumpster fire", "absolute L", 
            "massive L", "scam", "nightmare fuel", "cursed", "cancer", "toxic",
            "hot garbage", "cringe", "cope", "train wreck", "absolute joke",
            "not even mid", "below mid", "bottom tier", "hot mess", "dead",
            "flop", "flopped", "fraud", "waste of time", "atrocious",
            "straight trash", "unhinged", "delusional", "worst thing ever",
            "never touching this again", "avoid at all costs", "get your eyes checked"
        ]
        
        # High priority negative words that should have stronger influence on sentiment
        self.priority_negative_words = [
            "trash", "garbage", "terrible", "awful", "horrible", "hate", "disgusting",
            "useless", "waste", "pathetic", "worst", "dumpster fire", "massive L",
            "cringe", "toxic", "scam", "flop", "dogwater", "pure garbage"
        ]
        
        # High priority positive words
        self.priority_positive_words = [
            "amazing", "excellent", "perfect", "love", "incredible", "outstanding",
            "GOAT", "fire", "lit", "immaculate", "god tier", "absolutely goated", 
            "top tier", "legendary", "massive W", "certified fresh", "slaps"
        ]
        
        # Intensifiers that amplify the sentiment
        self.intensifiers = [
            "very", "extremely", "incredibly", "absolutely", "completely", "totally",
            "utterly", "thoroughly", "entirely", "fully", "really", "quite", "so",
            "too", "especially", "particularly", "exceptionally", "tremendously",
            "immensely", "hugely", "vastly", "insanely", "ridiculously", "seriously",
            "unbelievably", "amazingly", "remarkably", "decidedly", "highly",
            "profoundly", "deeply", "overwhelmingly", "purely", "super", "mega",
            "ultra", "extra", "beyond", "exceedingly", "extraordinarily",
            # New intensifiers from social media
            "af", "asf", "hella", "mad", "deadass", "literally", "literally just",
            "straight up", "straight", "actually", "low key", "high key", "fr",
            "frfr", "no cap", "actually", "100%", "absolutely", "definitely", 
            "without a doubt", "undeniably"
        ]
        
        # Common positive and negative hashtags
        self.positive_hashtags = [
            "love", "happy", "blessed", "grateful", "joy", "beautiful", "amazing", 
            "awesome", "excited", "wonderful", "success", "inspiration", "goals",
            "win", "winning", "blessed", "gratitude", "positive", "positivity",
            "goodvibes", "bestday", "perfect", "smile", "motivated", "thankful",
            "fun", "hope", "peace", "celebrate", "victory", "achievement", "proud",
            "supportive", "inspiring", "favorite", "wow", "gorgeous", "excellence",
            "champion", "yay", "congrats", "congratulations", "blessings", "goodnews",
            # New positive hashtags
            "vibes", "wholesome", "needthis", "trending", "viral", "impressive",
            "respected", "elite", "premium", "quality", "fresh", "stunning",
            "masterpiece", "aesthetic", "iconic", "slay", "mood", "obsessed"
        ]
        
        self.negative_hashtags = [
            "sad", "angry", "upset", "disappointed", "fail", "failure", "hate", 
            "depressed", "depression", "anxiety", "stressed", "tired", "exhausted",
            "worried", "heartbroken", "brokenheart", "lonely", "alone", "hurt",
            "pain", "tears", "crying", "frustrated", "mad", "annoyed", "rage",
            "badday", "disaster", "tragedy", "tragic", "disgusted", "disgusting",
            "horrible", "terrible", "awful", "ugh", "fml", "smh", "wtf", "outraged",
            "shame", "shameful", "pathetic", "ridiculous", "meaningless", "waste",
            "useless", "hardship", "struggle", "suffering", "victim", "trauma",
            # New negative hashtags
            "cringe", "cursed", "scam", "sus", "toxic", "redflags", "warning",
            "dontwasteyourtime", "disappointed", "notworth", "avoid", "staywoke",
            "dontbefooled", "hurtful", "triggered", "frustrated", "boycott",
            "cancelled", "problematic", "backlash", "scandal", "controversy",
            "dumpsterfire", "worstever", "flop", "eyeroll", "unfollowing"
        ]
        
        # Patterns for detecting mixed sentiment
        self.mixed_patterns = [
            r"but\s",
            r"however\s",
            r"although\s",
            r"though\s",
            r"while\s",
            r"despite\s",
            r"even though\s",
            r"on the other hand",
            r"mixed",
            r"nonetheless",
            r"nevertheless",
            r"yet\s",
            r"except\s",
            r"still\s.*but",
            r"otherwise",
            r"apart from",
            r"aside from",
            r"pros and cons",
            r"mixed feelings",
            r"good and bad",
            r"ups and downs",
            r"hit and miss",
            r"hit or miss"
        ]
        
        # Enhanced emoji lists with newer emojis
        # Positive emojis - updated with newer emojis
        self.positive_emojis = [
            # Traditional positive emojis
            "😀", "😃", "😄", "😁", "😆", "😊", "😇", "🙂", "😍", "🥰", "😘", "😗", "😙", "😚", 
            "😋", "😜", "😝", "😛", "🤗", "🤩", "🥳", "😺", "😸", "😻", "👍", "👏", "💪", "🙌", 
            "🎉", "✨", "💖", "❤️", "💕", "💞", "💓", "💗", "💙", "💚", "💛", "💜", "🧡", "🤍", 
            "🤎", "💯", "✔️", "😎", "😌",
            
            # Newer positive emojis
            "🔥", "💅", "✌️", "🤟", "🤙", "👊", "🫶", "🥹", "🤭", "🥺", "😌", "😮‍💨", "🥂",
            "🙏", "🤞", "✅", "💯", "🔝", "💫", "⭐", "🌟", "💎", "🏆", "🥇", "🎯", "✨", 
            "🍾", "🎊", "🎁", "❣️", "♥️", "🚀", "⚡", "📈", "🌻", "🌺", "🌈", "🌞", "🌠",
            "💰", "💪", "👌", "💅", "🌸", "🌹", "👑", "✅", "🫠", "❤️‍🔥", "🦾", "🫂",
            "😭🙌", "😩👌", "😌✨", "😫👏", "🤌", "🫴", "🤣👍", "😊💕", "🙏🔥"
        ]
        
        # Negative emojis - updated with newer emojis
        self.negative_emojis = [
            # Traditional negative emojis
            "😞", "😔", "😟", "😕", "🙁", "☹️", "😣", "😖", "😫", "😩", "🥺", "😢", "😭", "😤", 
            "😠", "😡", "🤬", "👎", "😾", "😿", "🙀", "😱", "😨", "😰", "😥", "😓", "😒",
            
            # Newer negative emojis
            "💔", "😬", "🙄", "😑", "😐", "😶", "😏", "🫤", "🫥", "😮‍💨", "🫠", "🤐", "🤕",
            "🤢", "🤮", "😷", "🥴", "😵", "🫣", "👀", "⚰️", "☠️", "💀", "🖕", "⛔", "🚫",
            "📉", "🗑️", "💩", "🤦‍♀️", "🤦‍♂️", "🤦", "🤷‍♀️", "🤷‍♂️", "🤷", "🫡", "🥱",
            "😮‍💨", "🙃", "😶‍🌫️", "😵‍💫", "😵", "🤯", "😳", "🚮", "🔪", "❌", "⛔",
            "👎🙄", "😠💢", "👋😒", "🤢🤮", "🤷‍♀️💀", "😭💔", "😞👎", "😡🚫"
        ]
        
        # Neutral emojis - updated with newer emojis
        self.neutral_emojis = [
            # Traditional neutral emojis
            "😐", "😑", "😶", "🤔", "😬", "🤨", "😏", "😒", "😕", "😯", "😲", "😳", "😦", 
            "😧", "😮", "😴", "🤤", "😪", "😵", "🤐", "🤢", "🤮", "😷", "🤒", "🤕", "🤑",
            
            # Newer neutral emojis
            "🫠", "🫣", "🫤", "🫡", "🫥", "😶‍💫", "❓", "❔", "⁉️", "‼️", "🤷", "🤔", "🧐",
            "👀", "👁️", "🗿", "💭", "🫦", "🕰️", "🧠", "🦧", "🫰", "🤌", "🦾", "🧘", "🧘‍♀️",
            "🧘‍♂️", "🕯️", "🧿", "🫙", "😬🤔", "👀👄👀", "🤨📸", "🧐🤷", "🚶‍♀️🚶‍♂️"
        ]

        # Special context-sensitive emojis
        self.context_sensitive_emojis = {
            # Emojis that change meaning based on context
            "😂": {
                "default": "positive",  # Default interpretation
                "positive_contexts": ["funny", "lol", "lmao", "hilarious", "joke", "haha", "laugh", "comedy", "humor"],
                "negative_contexts": ["fail", "dumb", "stupid", "ridiculous", "pathetic", "embarrassing", "sad", "crying"]
            },
            "💀": {
                "default": "negative",  # Default interpretation
                "positive_contexts": ["funny", "lol", "lmao", "hilarious", "joke", "dead", "im dead", "i'm dead", "killed me"],
                "negative_contexts": ["terrible", "awful", "embarrassing", "cringe", "bad", "fail"]
            },
            "😭": {
                "default": "negative",  # Default interpretation
                "positive_contexts": ["beautiful", "touching", "moved", "emotional", "love", "amazing", "perfect", "so good"],
                "negative_contexts": ["sad", "upset", "hurt", "painful", "depressing", "terrible", "awful"]
            },
            "😩": {
                "default": "negative",  # Default interpretation
                "positive_contexts": ["good", "amazing", "perfect", "so good", "delicious", "satisfying", "incredible"],
                "negative_contexts": ["exhausted", "tired", "frustrated", "annoying", "upset", "sad"]
            }
        }
        
        # --- Expanded mixed/ambiguous keywords and patterns ---
        self.mixed_keywords = [
            "bittersweet", "love-hate", "mixed feelings", "conflicted", "ambivalent", "torn", "double-edged",
            "mixed emotions", "uncertain", "on the fence", "gray area", "not sure how to feel", "both good and bad",
            "pros and cons", "mixed bag", "complicated", "hard to say", "not black and white", "not clear cut",
            "somewhat happy", "somewhat sad", "happy but sad", "sad but happy", "mixed review", "ambiguous",
            "unclear", "indecisive", "wishy-washy", "in two minds", "split feelings", "confusing", "paradoxical"
        ]
        # Always ensure mixed_keywords is initialized
        if not hasattr(self, 'mixed_keywords'):
            self.mixed_keywords = []
        self.mixed_patterns += [
            r"bittersweet", r"love[- ]hate", r"ambivalent", r"conflicted", r"torn", r"double[- ]edged",
            r"mixed emotions", r"uncertain", r"on the fence", r"gray area", r"not sure how to feel",
            r"complicated", r"not black and white", r"not clear cut", r"in two minds", r"split feelings",
            r"paradoxical"
        ]
        
        # Social media abbreviations and their sentiment implications
        self.social_media_abbreviations = {
            # Positive
            "gg": "positive",  # good game
            "ily": "positive",  # I love you
            "ilysm": "positive",  # I love you so much
            "goat": "positive",  # greatest of all time
            "lol": "positive",  # laughing out loud
            "lmao": "positive",  # laughing my ass off
            "rofl": "positive",  # rolling on floor laughing
            "tysm": "positive",  # thank you so much
            "lfg": "positive",  # let's f***ing go
            "fr": "neutral",    # for real (enhances whatever it's attached to)
            "frfr": "neutral",  # for real for real (stronger enhancer)
            "tbh": "neutral",   # to be honest
            "fwiw": "neutral",  # for what it's worth
            "imo": "neutral",   # in my opinion
            "imho": "neutral",  # in my humble opinion
            "idk": "neutral",   # I don't know
            "ngl": "neutral",   # not gonna lie
            
            # Negative
            "smh": "negative",  # shaking my head
            "fml": "negative",  # f*** my life
            "kms": "negative",  # kill myself (hyperbolic expression of frustration)
            "kys": "negative",  # kill yourself (highly negative)
            "stfu": "negative", # shut the f*** up
            "gtfo": "negative", # get the f*** out
            "ffs": "negative",  # for f***'s sake
            "wtf": "negative",  # what the f***
            "lmfao": "positive", # laughing my f***ing ass off (usually positive despite f-word)
            "istg": "negative",  # I swear to god (usually frustration)
            "pos": "negative",   # piece of s***
            "af": "neutral",     # as f*** (intensifier)
            "asf": "neutral",    # as f*** (intensifier)
            "bs": "negative",    # bulls***
            "ftw": "positive",   # for the win
            "goated": "positive", # greatest of all time (adjective form)
            "based": "positive",  # cool, agreeable
            "cap": "negative",    # lie/fake
            "no cap": "positive", # no lie/truth
            "cringe": "negative", # embarrassing
            "sus": "negative",    # suspicious
            "fomo": "negative",   # fear of missing out
            "salty": "negative",  # bitter/upset
            "slaps": "positive",  # excellent (music/food)
            "snack": "positive",  # attractive person
            "shook": "neutral",   # surprised
            "stan": "positive",   # big fan
            "thirsty": "negative", # desperate
            "yeet": "neutral",    # throw/discard
            "simp": "negative",   # overly attentive
            "L": "negative",      # loss/fail
            "W": "positive",      # win/success
            "ratio": "negative",  # getting more replies than likes (bad)
            "boomer": "negative", # old/out of touch
            "meme": "neutral",    # joke reference
            "mood": "positive",   # relatable feeling
            "slay": "positive",   # doing great
            "toxic": "negative",  # harmful behavior
            "vibe": "positive",   # feeling/atmosphere
            "woke": "neutral",    # socially aware
            "zaddy": "positive",  # attractive older man
            "bet": "positive",    # agreement/confirmation
            "bop": "positive",    # good song
            "fix": "negative",    # intoxicating substance
            "snatched": "positive", # looking good
            "receipts": "neutral", # evidence
            "triggered": "negative", # upset
            "yikes": "negative",   # expression of dismay
            "tfw": "neutral",      # that feeling when
            "mfw": "neutral",      # my face when
            "otp": "positive"      # one true pair
        }
        
        # Initialize cache for text analysis results
        self.analysis_cache = {}
        self.cache_max_size = 1000  # Maximum cache entries

        # --- Expanded critical/constructive keywords ---
        self.critical_keywords = [
            "constructive", "feedback", "suggest", "recommend improvement", "could be better", "room for improvement",
            "critique", "review", "advise", "recommendation", "constructively", "critical", "point out",
            "highlight issue", "highlight problem", "identify issue", "identify problem", "needs work",
            "could use improvement", "improve", "fix", "address", "recommend changes", "recommend fix",
            "constructive criticism", "constructive feedback", "constructive review", "constructive suggestion",
            "constructive advice", "constructive input", "constructive comment", "constructive remarks",
            "constructive points", "constructive note", "constructive observation"
        ]
    
    def detect_keywords(self, text, keyword_list):
        """Check if any of the keywords are in the text."""
        return any(keyword in text for keyword in keyword_list)
    
    def find_keywords(self, text, keyword_list):
        """Find all keywords from the list that appear in the text."""
        return [keyword for keyword in keyword_list if keyword in text]
    
    def has_mixed_sentiment(self, text):
        """Check if text has indicators of mixed sentiment."""
        return any(re.search(pattern, text) for pattern in self.mixed_patterns)
    
    def detect_intensifiers(self, text):
        """Detect intensifiers in the text and return the intensifier words found."""
        return [intensifier for intensifier in self.intensifiers if intensifier in text]
    
    def check_intensified_word(self, text, target_words):
        """Check if any target words are intensified in the text."""
        matches = []
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in target_words and i > 0 and words[i-1] in self.intensifiers:
                matches.append(f"{words[i-1]} {word}")
        
        # Check for intensifier followed by words (with possible intervening words)
        for intensifier in self.intensifiers:
            for word in target_words:
                pattern = r'{}.*{}'.format(re.escape(intensifier), re.escape(word))
                found_matches = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found_matches)
        
        return bool(matches), matches
        
    def extract_hashtags(self, text):
        """Extract hashtags from text and analyze their sentiment."""
        # Extract all hashtags from the text
        hashtags = re.findall(r'#(\w+)', text.lower())
        
        # Count positive and negative hashtags
        positive_count = 0
        negative_count = 0
        found_positive = []
        found_negative = []
        
        for tag in hashtags:
            # Check for positive hashtags
            if tag in self.positive_hashtags:
                positive_count += 1
                found_positive.append(tag)
            
            # Check for negative hashtags
            if tag in self.negative_hashtags:
                negative_count += 1
                found_negative.append(tag)
        
        # Generate sentiment score from hashtags (-1 to 1)
        hashtag_sentiment = 0
        if hashtags:
            hashtag_sentiment = (positive_count - negative_count) / len(hashtags)
        
        return {
            "hashtags": hashtags,
            "positive_hashtags": found_positive,
            "negative_hashtags": found_negative,
            "sentiment_score": hashtag_sentiment
        }
        
    def _normalize_emoji(self, emoji_char):
        """Remove variation selectors and skin tone modifiers from emoji for robust matching."""
        import unicodedata
        # Remove variation selectors (U+FE0F etc.) and skin tone modifiers (U+1F3FB - U+1F3FF)
        return ''.join(
            c for c in emoji_char
            if not (
                unicodedata.category(c) == 'Mn' or
                0x1F3FB <= ord(c) <= 0x1F3FF or
                ord(c) == 0xFE0F
            )
        )

    def _is_positive_crying_context(self, text):
        """Detect if the context around the crying emoji is positive (e.g., 'so happy', 'so funny')."""
        positive_crying_keywords = [
            "so happy", "so good", "so funny", "hilarious", "laughing", "amazing", "love", "best", "so cute", "adorable", "so proud", "so sweet", "so beautiful", "so touching", "i'm happy", "i am happy", "i'm so happy", "i'm so proud", "i'm so grateful", "i'm so thankful", "i'm so excited"
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in positive_crying_keywords)

    def _analyze_context_sensitive_emoji(self, emoji, text):
        """
        Analyze context-sensitive emojis based on surrounding text.
        
        Args:
            emoji: The emoji character to analyze
            text: The full text containing the emoji
            
        Returns:
            String: "positive", "negative", or "neutral" based on context
        """
        text_lower = text.lower()
        
        # Check if emoji is in our context-sensitive dictionary
        if emoji in self.context_sensitive_emojis:
            emoji_info = self.context_sensitive_emojis[emoji]
            
            # Check for positive context
            if any(context in text_lower for context in emoji_info["positive_contexts"]):
                return "positive"
                
            # Check for negative context
            if any(context in text_lower for context in emoji_info["negative_contexts"]):
                return "negative"
                
            # Default interpretation if no specific context is found
            return emoji_info["default"]
        
        # Return None for emojis not needing special context handling
        return None

    def extract_emojis(self, text):
        """Extract emojis from text and analyze their sentiment with enhanced context analysis."""
        try:
            import emoji
        except ImportError:
            return {"emojis": [], "positive_count": 0, "negative_count": 0, "neutral_count": 0, "sentiment_score": 0}
            
        # Check cache first for performance
        if text in self.analysis_cache and "emoji_analysis" in self.analysis_cache[text]:
            return self.analysis_cache[text]["emoji_analysis"]
            
        found_emojis = emoji.emoji_list(text)
        emoji_chars = [item["emoji"] for item in found_emojis]
        normalized_emojis = [self._normalize_emoji(e) for e in emoji_chars]
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for orig, norm in zip(emoji_chars, normalized_emojis):
            # First check if this is a context-sensitive emoji
            context_sentiment = self._analyze_context_sensitive_emoji(norm, text)
            
            if context_sentiment == "positive":
                positive_count += 1
            elif context_sentiment == "negative":
                negative_count += 1
            elif context_sentiment == "neutral":
                neutral_count += 1
            # If not context-sensitive, use standard emoji lists
            elif norm in self.positive_emojis:
                positive_count += 1
            elif norm in self.negative_emojis:
                negative_count += 1
            elif norm in self.neutral_emojis:
                neutral_count += 1
        
        emoji_sentiment = 0
        total = positive_count + negative_count + neutral_count
        if total:
            emoji_sentiment = (positive_count - negative_count) / total
        
        result = {
            "emojis": emoji_chars,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "sentiment_score": emoji_sentiment
        }
        
        # Cache the result
        if text not in self.analysis_cache:
            self.analysis_cache[text] = {}
        self.analysis_cache[text]["emoji_analysis"] = result
        
        # Manage cache size
        if len(self.analysis_cache) > self.cache_max_size:
            # Remove oldest item (assuming dict maintains insertion order in Python 3.7+)
            self.analysis_cache.pop(next(iter(self.analysis_cache)))
        
        return result

    def has_negation(self, text):
        """Detect negation phrases that might flip the sentiment."""
        negation_phrases = [
            "not ", "n't ", "never ", "no ", "none ", "neither ", "nor ",
            "doesn't ", "don't ", "didn't ", "isn't ", "aren't ",
            "wasn't ", "weren't ", "hardly ", "scarcely ", "barely ",
            "couldn't ", "can't ", "won't ", "wouldn't ", "shouldn't ", "haven't ",
            "hasn't ", "hadn't ", "without ", "lacking "
        ]
        return any(phrase in text for phrase in negation_phrases)

    def get_raw_score(self, text):
        """
        Get a more granular raw sentiment score by combining the model's score with additional heuristics.
        Returns a float between 0 and 1, where 0 is negative and 1 is positive.
        """
        # Get base model score
        result = self.analyzer(text)[0]
        if result['label'] == 'POSITIVE':
            base_score = result['score']
        else:
            base_score = 1 - result['score']

        # Heuristic adjustments
        text_lower = text.lower()
        emoji_info = self.extract_emojis(text)
        hashtag_info = self.extract_hashtags(text)
        intensifier_score = self._intensifier_score(text)
        domain_pos, domain_neg = self._domain_specific_keywords(text)
        sarcasm = self._detect_sarcasm(text)
        has_negation = self.has_negation(text)

        # Emoji adjustment
        emoji_adj = emoji_info['sentiment_score'] * 0.1  # up to ±0.1
        # Hashtag adjustment
        hashtag_adj = hashtag_info['sentiment_score'] * 0.1  # up to ±0.1
        # Intensifier adjustment
        intensifier_adj = (intensifier_score - 1.0) * 0.1  # up to +0.05
        # Domain-specific adjustment
        domain_adj = 0.05 if domain_pos else -0.05 if domain_neg else 0
        # Sarcasm flips the base score
        sarcasm_adj = -0.2 if sarcasm else 0
        # Negation adjustment
        negation_adj = -0.1 if has_negation else 0

        # Combine all adjustments
        score = base_score + emoji_adj + hashtag_adj + intensifier_adj + domain_adj + sarcasm_adj + negation_adj
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        return score

    def _analyze_emoji_sequence(self, emoji_chars, text):
        """Analyze consecutive emoji sequences for combined sentiment."""
        # Example: 😂😭 or 😭❤️
        if not emoji_chars:
            return 0
        # Map each emoji to sentiment: 1=positive, -1=negative, 0=neutral
        sentiments = []
        for e in emoji_chars:
            norm = self._normalize_emoji(e)
            if norm in self.positive_emojis:
                sentiments.append(1)
            elif norm in self.negative_emojis:
                sentiments.append(-1)
            elif norm in self.neutral_emojis:
                sentiments.append(0)
        # If both positive and negative in sequence, mark as mixed (neutral)
        if 1 in sentiments and -1 in sentiments:
            return 0
        # If all positive or all negative
        if all(s == 1 for s in sentiments):
            return 1
        if all(s == -1 for s in sentiments):
            return -1
        # Otherwise, sum up
        return sum(sentiments)

    def _detect_sarcasm(self, text):
        """Detect sarcasm/irony using patterns and sarcastic emojis."""
        sarcasm_patterns = [
            r"yeah, right", r"as if", r"sure thing", r"just great", r"thanks a lot", r"nice job", r"good luck with that",
            r"can't wait", r"what a surprise", r"oh great", r"love that for me", r"i'm so sure", r"wonderful", r"amazing", r"awesome"
        ]
        sarcastic_emojis = ["🙄", "😒", "😏", "😑", "😬", "😹"]
        if any(re.search(p, text, re.IGNORECASE) for p in sarcasm_patterns):
            return True
        if any(e in text for e in sarcastic_emojis):
            return True
        return False

    def _negation_scope(self, text, keywords):
        """Check if negation directly precedes a keyword (e.g., 'not good')."""
        negations = ["not", "never", "no", "n't", "hardly", "barely", "scarcely"]
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in keywords and i > 0 and words[i-1] in negations:
                return True
        return False

    def _intensifier_score(self, text):
        """Detect exclamation marks, all-caps, repeated letters/emojis to intensify sentiment."""
        score = 1.0
        if text.count("!") >= 2:
            score += 0.2
        if re.search(r"[A-Z]{3,}", text):
            score += 0.2
        if re.search(r"([a-zA-Z])\1{2,}", text):
            score += 0.2
        if re.search(r"([\U0001F600-\U0001F64F])\1{1,}", text):
            score += 0.2
        return min(score, 1.5)

    def _domain_specific_keywords(self, text):
        """Placeholder for domain-specific keywords/slang."""
        domain_positive = ["lit", "fire", "on point", "slaps", "goat", "banger"]
        domain_negative = ["cringe", "sus", "mid", "flop", "trash tier"]
        found_pos = any(w in text for w in domain_positive)
        found_neg = any(w in text for w in domain_negative)
        return found_pos, found_neg

    def has_mixed_signals(self, text, found_priority_negative, found_priority_positive, has_positive_hashtags, has_negative_hashtags, has_positive_emojis, has_negative_emojis):
        """Detect if both strong positive and negative signals are present."""
        strong_pos = bool(found_priority_positive or has_positive_hashtags or has_positive_emojis)
        strong_neg = bool(found_priority_negative or has_negative_hashtags or has_negative_emojis)
        return strong_pos and strong_neg

    def detect_critical_keywords(self, text):
        """Check if any critical/constructive keywords are in the text."""
        return any(keyword in text for keyword in self.critical_keywords)

    def detect_social_media_abbreviations(self, text):
        """
        Detect and analyze social media abbreviations/slang for sentiment implications.
        
        Args:
            text: The text to analyze for social media abbreviations
            
        Returns:
            Dict containing detected abbreviations and their sentiment impact
        """
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Check for multi-word abbreviations first (like "no cap")
        multi_word_abbrevs = ["no cap", "low key", "high key", "for real"]
        
        found_abbreviations = {}
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        # Check for multi-word abbreviations in text
        for abbrev in multi_word_abbrevs:
            if abbrev in text.lower():
                sentiment = self.social_media_abbreviations.get(abbrev.replace(" ", ""), "neutral")
                found_abbreviations[abbrev] = sentiment
                if sentiment == "positive":
                    positive_count += 1
                elif sentiment == "negative":
                    negative_count += 1
                else:
                    neutral_count += 1
        
        # Check for single word abbreviations
        for word in words:
            if word in self.social_media_abbreviations:
                sentiment = self.social_media_abbreviations[word]
                found_abbreviations[word] = sentiment
                if sentiment == "positive":
                    positive_count += 1
                elif sentiment == "negative":
                    negative_count += 1
                else:
                    neutral_count += 1
        
        # Calculate overall sentiment impact
        sentiment_score = 0
        total = positive_count + negative_count
        if total > 0:
            sentiment_score = (positive_count - negative_count) / total
        
        result = {
            "abbreviations": found_abbreviations,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "sentiment_score": sentiment_score,
            "has_abbreviations": len(found_abbreviations) > 0
        }
        
        return result

    def detect_hateful_content(self, text):
        """Detect hateful or offensive language in the text."""
        hateful_keywords = [
            "hate", "racist", "sexist", "bigot", "offensive", "slur", "discrimination",
            "violence", "abuse", "harassment", "intolerant", "prejudice", "xenophobia",
            "homophobia", "transphobia", "misogyny", "misandry", "hateful"
        ]
        return any(keyword in text.lower() for keyword in hateful_keywords)

    def predict(self, texts):
        """
        Analyze sentiment and return predictions.
        Args:
            texts: A string or list of strings to analyze
        Returns:
            List of sentiment category indices (0-2) corresponding to the 3 sentiment levels
        """
        # Make sure input is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Get predictions from the pipeline
        results = self.analyzer(texts)
        
        # Convert the pipeline output to our 3 sentiment categories
        predictions = []
        for i, result in enumerate(results):
            # Get the positive score (confidence level)
            label = result['label']
            score = result['score']
            text = texts[i].lower()
            
            # Initialize debug info dictionary
            debug_info = {
                "label": label,
                "score": f"{score:.4f}",
                "features": []
            }
            
            # Check for keyword presence
            has_very_negative = self.detect_keywords(text, self.very_negative_keywords)
            has_negative = self.detect_keywords(text, self.negative_keywords)
            has_neutral = self.detect_keywords(text, self.neutral_keywords)
            has_positive = self.detect_keywords(text, self.positive_keywords)
            has_very_positive = self.detect_keywords(text, self.very_positive_keywords)
            has_mixed = self.has_mixed_sentiment(text) or self.detect_keywords(text, self.mixed_keywords)
            has_critical = self.detect_critical_keywords(text)
            
            # Check for hashtags
            hashtag_info = self.extract_hashtags(text)
            has_positive_hashtags = len(hashtag_info["positive_hashtags"]) > 0
            has_negative_hashtags = len(hashtag_info["negative_hashtags"]) > 0
            hashtag_sentiment = hashtag_info["sentiment_score"]
            
            # Check for priority words and intensifiers
            found_priority_negative = self.find_keywords(text, self.priority_negative_words)
            found_priority_positive = self.find_keywords(text, self.priority_positive_words)
            found_intensifiers = self.detect_intensifiers(text)
            
            # Check for intensified priority words
            has_intensified_negative, intensified_negative_matches = self.check_intensified_word(text, self.priority_negative_words)
            has_intensified_positive, intensified_positive_matches = self.check_intensified_word(text, self.priority_positive_words)
            
            # Add detection results to debug info
            if has_very_negative:
                debug_info["features"].append("very negative keywords")
            if has_negative:
                debug_info["features"].append("negative keywords")
            if has_neutral:
                debug_info["features"].append("neutral keywords")
            if has_positive:
                debug_info["features"].append("positive keywords")
            if has_very_positive:
                debug_info["features"].append("very positive keywords")
            if has_mixed:
                debug_info["features"].append("mixed sentiment indicators")
            if hashtag_info["hashtags"]:
                debug_info["features"].append(f"hashtags: {', '.join(hashtag_info['hashtags'])}")
            if has_positive_hashtags:
                debug_info["features"].append(f"positive hashtags: {', '.join(hashtag_info['positive_hashtags'])}")
            if has_negative_hashtags:
                debug_info["features"].append(f"negative hashtags: {', '.join(hashtag_info['negative_hashtags'])}")
            if found_priority_negative:
                debug_info["features"].append(f"priority negative words: {', '.join(found_priority_negative)}")
            if found_priority_positive:
                debug_info["features"].append(f"priority positive words: {', '.join(found_priority_positive)}")
            if found_intensifiers:
                debug_info["features"].append(f"intensifiers: {', '.join(found_intensifiers)}")
            if has_intensified_negative:
                debug_info["features"].append(f"intensified negative words: {', '.join([str(m) for m in intensified_negative_matches])}")
            if has_intensified_positive:
                debug_info["features"].append(f"intensified positive words: {', '.join([str(m) for m in intensified_positive_matches])}")
            
            emoji_info = self.extract_emojis(text)
            has_positive_emojis = emoji_info["positive_count"] > 0
            has_negative_emojis = emoji_info["negative_count"] > 0
            emoji_sentiment = emoji_info["sentiment_score"]
            emoji_sequence_sentiment = self._analyze_emoji_sequence(emoji_info["emojis"], text)
            sarcasm = self._detect_sarcasm(text)
            neg_scope_pos = self._negation_scope(text, self.positive_keywords)
            neg_scope_neg = self._negation_scope(text, self.negative_keywords)
            intensifier = self._intensifier_score(text)
            domain_pos, domain_neg = self._domain_specific_keywords(text)
            has_negation = self.has_negation(text)
            
            # --- Mixed/ambiguous logic ---
            if self.has_mixed_signals(text, found_priority_negative, found_priority_positive, has_positive_hashtags, has_negative_hashtags, has_positive_emojis, has_negative_emojis):
                category = 1  # neutral/mixed
                debug_info["features"].append("strong mixed signals (positive & negative)")
            elif has_mixed:
                category = 1
                debug_info["features"].append("mixed/ambiguous keywords or patterns")
            # --- Critical/constructive logic ---
            elif has_critical:
                # If critical/constructive, lean toward neutral or negative, but boost weight
                if has_negative or found_priority_negative:
                    category = 0  # negative
                else:
                    category = 1  # neutral/constructive
                debug_info["features"].append("critical/constructive keywords detected (weighted)")
            else:
                # Advanced sentiment classification rules
                if has_intensified_negative or (found_priority_negative and found_intensifiers):
                    # Strongly prioritize intensified negative words like "incredibly trash"
                    category = 0 if not has_negation else 2
                elif has_intensified_positive or (found_priority_positive and found_intensifiers):
                    # Strongly prioritize intensified positive words
                    category = 2 if not has_negation else 0
                elif found_priority_negative:
                    # Prioritize high-impact negative words even without intensifiers
                    category = 0 if len(found_priority_negative) > 1 else (0 if not has_negation else 2)
                elif found_priority_positive:
                    # Prioritize high-impact positive words even without intensifiers
                    category = 2 if len(found_priority_positive) > 1 else (2 if not has_negation else 0)
                elif has_mixed:
                    # Mixed sentiment typically indicates a neutral overall sentiment
                    category = 1
                elif has_negation:
                    if label == "POSITIVE":
                        category = 0
                    else:
                        category = 2
                elif label == "NEGATIVE":
                    negative_criteria = [
                        has_very_negative,
                        has_negative,
                        has_negative_hashtags and len(hashtag_info["negative_hashtags"]) > 0,
                        has_negative_emojis and emoji_info["negative_count"] > 0,
                        score > 0.9
                    ]
                    if label == "NEGATIVE" and sum(1 for c in negative_criteria if c) >= 2:
                        category = 0  # negative
                    if has_very_negative or (has_negative_hashtags and hashtag_sentiment < -0.5) or (has_negative_emojis and emoji_sentiment < -0.5):
                        category = 0  # negative
                    elif has_negative or (has_negative_hashtags) or (has_negative_emojis) or score > 0.9:
                        category = 0  # negative
                    elif has_neutral:
                        category = 1  # neutral
                    else:
                        # Lower confidence negative statements or without strong keywords
                        category = 0  # default to negative for NEGATIVE label
                else:  # label is POSITIVE
                    if has_very_positive or (has_positive_hashtags and hashtag_sentiment > 0.5) or (has_positive_emojis and emoji_sentiment > 0.5):
                        category = 2  # positive
                    elif has_positive or (has_positive_hashtags) or (has_positive_emojis) or score > 0.9:
                        category = 2  # positive
                    elif has_neutral:
                        category = 1  # neutral
                    else:
                        # Lower confidence positive statements or without strong keywords
                        category = 2  # default to positive for POSITIVE label
                
                # Check for neutral overrides - these take precedence unless we have priority words
                if not (found_priority_negative or found_priority_positive):
                    if (len(text.split()) <= 3) and has_neutral:  # Short neutral statements
                        category = 1
                    elif has_neutral and not (has_very_positive or has_very_negative):
                        if not (has_positive and score > 0.95) and not (has_negative and score > 0.95):
                            # Don't override if we have strong hashtag signals
                            if not (has_positive_hashtags and hashtag_sentiment > 0.5) and not (has_negative_hashtags and hashtag_sentiment < -0.5):
                                category = 1
                
                # Hashtag override - strong hashtag signals can override weak model predictions
                if hashtag_info["hashtags"]:
                    if hashtag_sentiment > 0.7 and category < 2:  # Strong positive hashtags
                        category = 2  # Bump to at least positive
                    elif hashtag_sentiment < -0.7 and category > 0:  # Strong negative hashtags
                        category = 0  # Bump down to at least negative
                
                # Emoji override
                if emoji_info["emojis"] and len(emoji_info["emojis"]) > 0:
                    if emoji_sentiment > 0.8 and category < 2:
                        category = 2
                    elif emoji_sentiment > 0.6 and category < 2:
                        category = 2
                    elif emoji_sentiment < -0.8 and category > 0:
                        category = 0
                    elif emoji_sentiment < -0.6 and category > 0:
                        category = 0
                
                # Sarcasm flips sentiment
                if sarcasm:
                    if label == "POSITIVE":
                        category = 0
                    else:
                        category = 2
                    predictions.append(category)
                    continue
                # Emoji sequence override
                if emoji_sequence_sentiment > 0 and category < 2:
                    category = 2
                elif emoji_sequence_sentiment < 0 and category > 0:
                    category = 0
                # Negation scope override
                if neg_scope_pos:
                    category = 0
                if neg_scope_neg:
                    category = 2
                # Intensifier adjustment
                if intensifier > 1.0:
                    if category == 2:
                        category = 2
                    elif category == 0:
                        category = 0
                # Domain-specific lexicon
                if domain_pos and category < 2:
                    category = 2
                if domain_neg and category > 0:
                    category = 0
            
            predictions.append(category)
            
            # Print debug information
            sentiment_label = self.get_sentiment_label(category)
            print(f"'{texts[i]}' → {sentiment_label}")
            print(f"  Label: {debug_info['label']}, Score: {debug_info['score']}")
            if hashtag_info["hashtags"]:
                print(f"  Hashtag sentiment: {hashtag_sentiment:.2f}")
            if debug_info["features"]:
                print(f"  Features: {', '.join(debug_info["features"])}")
        
        return predictions
    
    def get_sentiment_label(self, category_index):
        """Get the text label for a sentiment category index."""
        return self.sentiment_categories[category_index]

    def batch_predict(self, texts_list, batch_size=16):
        """
        Process multiple texts in efficient batches for improved performance.
        
        Args:
            texts_list: List of strings to analyze
            batch_size: Number of texts to process in each batch
            
        Returns:
            Dictionary with sentiment predictions and processing statistics
        """
        if not texts_list:
            return {"predictions": [], "processing_time": 0, "texts_processed": 0}
        
        import time
        start_time = time.time()
        
        all_predictions = []
        total_texts = len(texts_list)
        batches = [texts_list[i:i + batch_size] for i in range(0, total_texts, batch_size)]
        
        for batch in batches:
            # Process each batch
            batch_predictions = self.predict(batch)
            all_predictions.extend(batch_predictions)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        results = {
            "predictions": all_predictions,
            "prediction_categories": [self.get_sentiment_label(pred) for pred in all_predictions],
            "processing_time": processing_time,
            "texts_processed": total_texts,
            "texts_per_second": total_texts / processing_time if processing_time > 0 else 0
        }
        
        return results

    def analyze_sentiment_trends(self, texts_list, user_info=None, timestamps=None):
        """
        Analyze sentiment trends across multiple texts, optionally with timestamps and user context.
        
        Args:
            texts_list: List of strings to analyze
            user_info: Optional dict with user metadata
            timestamps: Optional list of datetime objects corresponding to the texts
            
        Returns:
            Dict with trend analysis and aggregate statistics
        """
        # First get all sentiment predictions
        batch_results = self.batch_predict(texts_list)
        predictions = batch_results["predictions"]
        
        if not predictions:
            return {"error": "No texts to analyze"}
        
        # Calculate distribution
        sentiment_distribution = {i: predictions.count(i) for i in range(3)}
        total_texts = len(predictions)
        sentiment_percentages = {
            self.get_sentiment_label(k): round((v / total_texts) * 100, 1) 
            for k, v in sentiment_distribution.items()
        }
        
        # Calculate overall sentiment score (-100 to 100 scale)
        # Weight: negative = -1, neutral = 0, positive = 1
        weights = {0: -1, 1: 0, 2: 1}
        weighted_sum = sum(weights[pred] * 50 for pred in predictions)  # Scale to -100 to 100
        overall_score = weighted_sum / total_texts
        
        # Detect sentiment shifts (if timestamps provided)
        sentiment_shifts = []
        if timestamps and len(timestamps) == len(predictions):
            # Sort predictions by timestamp
            sorted_data = sorted(zip(timestamps, predictions))
            # Look for significant shifts
            for i in range(1, len(sorted_data)):
                prev_sentiment = sorted_data[i-1][1]
                curr_sentiment = sorted_data[i][1]
                # If sentiment changed by 2 or more levels
                if abs(curr_sentiment - prev_sentiment) >= 2:
                    sentiment_shifts.append({
                        "timestamp": sorted_data[i][0],
                        "from": self.get_sentiment_label(prev_sentiment),
                        "to": self.get_sentiment_label(curr_sentiment),
                        "shift_magnitude": abs(curr_sentiment - prev_sentiment)
                    })
        
        # Get dominant emotion words
        positive_words = []
        negative_words = []
        for text, pred in zip(texts_list, predictions):
            if pred == 2:  # Positive
                # Extract positive keywords
                text_lower = text.lower()
                found_positive = [word for word in self.very_positive_keywords + self.positive_keywords 
                                 if word in text_lower]
                positive_words.extend(found_positive)
            elif pred == 0:  # Negative
                # Extract negative keywords
                text_lower = text.lower()
                found_negative = [word for word in self.very_negative_keywords + self.negative_keywords 
                                 if word in text_lower]
                negative_words.extend(found_negative)
        
        # Count word frequencies
        from collections import Counter
        positive_word_counts = Counter(positive_words)
        negative_word_counts = Counter(negative_words)
        
        # Get top words
        top_positive = positive_word_counts.most_common(5)
        top_negative = negative_word_counts.most_common(5)
        
        # Prepare the analysis results
        analysis = {
            "overall_score": overall_score,
            "sentiment_distribution": {
                "counts": sentiment_distribution,
                "percentages": sentiment_percentages
            },
            "dominant_sentiment": self.get_sentiment_label(max(sentiment_distribution, key=sentiment_distribution.get)),
            "sentiment_keywords": {
                "positive": [{"word": word, "count": count} for word, count in top_positive],
                "negative": [{"word": word, "count": count} for word, count in top_negative]
            },
            "processing_stats": {
                "texts_processed": total_texts,
                "processing_time": batch_results["processing_time"],
                "texts_per_second": batch_results["texts_per_second"]
            }
        }
        
        # Include sentiment shifts if available
        if sentiment_shifts:
            analysis["sentiment_shifts"] = sentiment_shifts
        
        return analysis

    def detect_topics(self, texts_list):
        """
        Detect common topics or themes across multiple texts using keyword extraction.
        
        Args:
            texts_list: List of strings to analyze
            
        Returns:
            Dict containing detected topics and their frequency
        """
        # Topic keywords by domain
        topic_categories = {
            "politics": [
                "government", "election", "vote", "democracy", "president", "minister", 
                "parliament", "political", "policy", "campaign", "party", "republican", 
                "democrat", "congress", "senator", "liberal", "conservative", "law", 
                "legislation", "rights", "freedom"
            ],
            "conflict": [
                "war", "conflict", "battle", "violence", "geopolitics", "diplomacy", "peace talks",
                "ceasefire", "military", "invasion", "occupation", "civil war", "insurgency",
                "guerrilla", "terrorism", "extremism", "militant", "rebellion", "resistance",
                "airstrike", "bombardment", "artillery", "combat", "hostilities", "offensive",
                "peacekeeping", "demilitarized zone", "siege", "armistice", "treaty", "atrocity",
                "humanitarian crisis", "refugee crisis", "civilian casualties", "collateral damage",
                # New keywords
                "armed conflict", "proxy war", "ethnic cleansing", "genocide", "massacre", 
                "human rights abuse", "torture", "disarmament", "sanctions", "aggression",
                "intervention", "war crimes", "casualties", "escalation", "truce",
                "no-fly zone", "buffer zone", "frontline", "warzone", "troops", "deployment",
                "guerrilla warfare", "insurgent", "insurrection", "coup", "revolution",
                "uprising", "counterinsurgency", "raid", "assault", "hostage", "militia"
            ],
            "regional_tensions": [
                "kashmir", "india", "pakistan", "border dispute", "ceasefire", "pahalgam", "tourism",
                "hindu", "muslim", "religious conflict", "communal violence", "riots", "sectarianism",
                "ukraine", "russia", "crimea", "donbas", "nato", "putin", "zelenskyy",
                "israel", "gaza", "palestine", "west bank", "hamas", "idf", "hezbollah", "netanyahu",
                "china", "taiwan", "south china sea", "trade war", "xi jinping", "territorial dispute",
                "north korea", "south korea", "dmz", "nuclear test", "missile launch",
                "myanmar", "rohingya", "ethnic cleansing", "junta", "military coup",
                "afghanistan", "taliban", "isis", "al-qaeda", "us withdrawal",
                # New keywords
                "territorial integrity", "sovereignty", "separatist", "secessionist", "autonomy",
                "self-determination", "ethnic minority", "religious persecution", "forced migration",
                "refugee camp", "internally displaced", "stateless", "border skirmish",
                "demarcation line", "disputed territory", "annexation", "occupation",
                "balkanization", "ethnic enclave", "buffer state", "proxy state",
                "cultural genocide", "ethnic tensions", "religious extremism", "radicalization",
                "interfaith conflict", "sunni", "shia", "orthodox", "fundamentalist", "militant"
            ],
            "economic_issues": [
                "inflation", "recession", "economic downturn", "price hike", "cost of living",
                "financial crisis", "market crash", "depression", "stagflation", "hyperinflation",
                "unemployment", "layoffs", "job losses", "austerity", "bailout", "debt crisis",
                "budget deficit", "economic sanctions", "trade embargo", "currency devaluation",
                "stock market", "bear market", "bull market", "monetary policy", "interest rates",
                "federal reserve", "central bank", "supply chain", "shortage", "rationing",
                # New keywords
                "economic collapse", "bankruptcy", "insolvency", "credit crunch", "liquidity crisis",
                "sovereign debt", "bond yields", "fiscal policy", "taxation", "tax evasion",
                "capital flight", "foreign investment", "divestment", "economic inequality", 
                "wealth gap", "poverty line", "subsistence", "food insecurity", "housing crisis",
                "homelessness", "wage stagnation", "minimum wage", "labor rights", "outsourcing",
                "offshoring", "commodity prices", "energy crisis", "fuel shortage", "price gouging",
                "black market", "informal economy", "underground economy", "economic sanctions"
            ],
            "controversy": [
                "controversy", "scandal", "dispute", "allegation", "backlash", "boycott",
                "outrage", "uproar", "protest", "demonstration", "petition", "criticism",
                "accusation", "expose", "whistleblower", "cover-up", "conspiracy", "corruption",
                "bribery", "embezzlement", "fraud", "misconduct", "impeachment", "resignation",
                "lawsuit", "legal battle", "defamation", "slander", "libel", "censorship",
                "ban", "cancellation", "deplatforming", "disinformation", "fake news",
                # New keywords
                "leaked documents", "sex scandal", "ethics violation", "conflict of interest",
                "insider trading", "money laundering", "tax evasion", "nepotism", "cronyism",
                "political bias", "media bias", "partisan", "polarization", "culture war",
                "extremist views", "hate speech", "incitement", "propaganda", "misinformation",
                "conspiracy theory", "denial", "revisionism", "public outrage", "moral panic",
                "call-out", "accountability", "apology demand", "public relations crisis",
                "damage control", "reputation management", "gag order", "non-disclosure"
            ],
            "technology": [
                "tech", "technology", "software", "hardware", "app", "device", "update", 
                "algorithm", "computer", "smartphone", "digital", "innovation", "ai", "data",
                "programming", "code", "internet", "mobile", "web", "online", "cyber"
            ],
            "entertainment": [
                "movie", "film", "series", "show", "music", "song", "album", "artist", "actor",
                "celebrity", "concert", "performance", "streaming", "netflix", "tv", "television",
                "game", "gaming", "play", "theater", "cinema", "festival", "entertainment"
            ],
            "sports": [
                "game", "team", "player", "win", "lose", "match", "tournament", "championship",
                "league", "score", "ball", "coach", "stadium", "athlete", "sport", "football",
                "soccer", "basketball", "baseball", "hockey", "tennis", "golf", "olympics"
            ],
            "health": [
                "health", "medical", "doctor", "hospital", "medication", "treatment", "symptom", 
                "disease", "diagnosis", "therapy", "mental", "physical", "wellness", "fitness",
                "exercise", "diet", "nutrition", "healthcare", "vaccine", "pandemic", "virus"
            ],
            "business": [
                "business", "company", "market", "product", "service", "customer", "startup",
                "investment", "economy", "economic", "stock", "price", "financial", "finance",
                "money", "profit", "revenue", "sales", "brand", "marketing", "entrepreneur"
            ],
            "social_issues": [
                "social", "community", "society", "issue", "justice", "equality", "inequality",
                "discrimination", "racism", "sexism", "protest", "movement", "activism", "rights",
                "gender", "identity", "climate", "environment", "poverty", "education", "reform"
            ],
            "personal": [
                "feel", "feeling", "emotion", "experience", "life", "personal", "self", "myself",
                "family", "friend", "relationship", "love", "hate", "happy", "sad", "angry", 
                "excited", "disappointed", "memory", "thought", "belief", "opinion"
            ]
        }
        
        # Initialize counter for each topic
        topic_counts = {topic: 0 for topic in topic_categories}
        
        # Process each text
        for text in texts_list:
            text_lower = text.lower()
            
            # Analyze each topic category
            for topic, keywords in topic_categories.items():
                # Count how many keywords from this topic appear in the text
                matches = [keyword for keyword in keywords if keyword in text_lower]
                if matches:
                    topic_counts[topic] += 1
        
        # Calculate percentages
        total_texts = len(texts_list)
        topic_percentages = {}
        for topic, count in topic_counts.items():
            if total_texts > 0:
                topic_percentages[topic] = round((count / total_texts) * 100, 1)
            else:
                topic_percentages[topic] = 0
        
        # Find specific keywords that appeared most frequently
        all_keywords = []
        for topic, keywords in topic_categories.items():
            for keyword in keywords:
                # Count occurrences across all texts
                keyword_count = sum(1 for text in texts_list if keyword in text.lower())
                if keyword_count > 0:
                    all_keywords.append((keyword, keyword_count))
        
        # Sort and get top keywords
        all_keywords.sort(key=lambda x: x[1], reverse=True)
        top_keywords = all_keywords[:10]  # Top 10 keywords
        
        # Determine primary and secondary topics
        sorted_topics = sorted(topic_percentages.items(), key=lambda x: x[1], reverse=True)
        primary_topics = [topic for topic, percentage in sorted_topics if percentage > 20]
        secondary_topics = [topic for topic, percentage in sorted_topics if 5 <= percentage <= 20 and topic not in primary_topics]
        
        result = {
            "primary_topics": primary_topics,
            "secondary_topics": secondary_topics,
            "topic_percentages": topic_percentages,
            "top_keywords": [{"keyword": kw, "count": count} for kw, count in top_keywords]
        }
        
        return result

    def prepare_visualization_data(self, texts_list, timestamps=None):
        """
        Prepare data structures suitable for sentiment visualization in the browser extension.
        
        Args:
            texts_list: List of strings to analyze
            timestamps: Optional list of datetime objects corresponding to the texts
            
        Returns:
            Dict with formatted data for visualization (charts, graphs, etc.)
        """
        # Get sentiment predictions
        batch_results = self.batch_predict(texts_list)
        predictions = batch_results["predictions"]
        
        if not predictions:
            return {"error": "No texts to analyze"}
        
        # Basic sentiment distribution for pie/donut chart
        sentiment_counts = {i: predictions.count(i) for i in range(3)}
        sentiment_labels = [self.get_sentiment_label(i) for i in range(3)]
        sentiment_data = {
            "labels": sentiment_labels,
            "counts": [sentiment_counts.get(i, 0) for i in range(3)],
            "colors": [
                "#e74c3c",  # negative (red)
                "#7f8c8d",  # neutral (gray)
                "#2ecc71"   # positive (green)
            ]
        }
        
        # Time series data if timestamps are provided
        time_series_data = None
        if timestamps and len(timestamps) == len(predictions):
            # Sort by timestamp
            time_data = sorted(zip(timestamps, predictions))
            time_series_data = {
                "timestamps": [str(t) for t, _ in time_data],
                "values": [p for _, p in time_data],
                "labels": [self.get_sentiment_label(p) for _, p in time_data]
            }
        
        # Prepare final visualization data package
        visualization_data = {
            "sentiment_distribution": sentiment_data,
            "processing_stats": {
                "texts_processed": len(texts_list),
                "processing_time": batch_results["processing_time"]
            }
        }
        
        if time_series_data:
            visualization_data["time_series"] = time_series_data
        
        return visualization_data

    def learn_from_feedback(self, text, original_prediction, corrected_prediction, save_feedback=True):
        """
        Learn from user feedback on sentiment predictions to improve future analyses.
        
        Args:
            text: The text that was analyzed
            original_prediction: The original prediction category (0-2)
            corrected_prediction: The corrected prediction category (0-2) from user
            save_feedback: Whether to save feedback to disk for future model training
            
        Returns:
            Dict with learning results and improvement suggestions
        """
        if original_prediction == corrected_prediction:
            return {"success": False, "message": "No correction needed, predictions match"}
        
        # Extract features from the text
        text_lower = text.lower()
        words = text_lower.split()
        
        # Analyze what might have caused the misclassification
        # 1. Check for negations that might have been missed
        negated = self.has_negation(text_lower)
        
        # 2. Check for social media slang
        slang_info = self.detect_social_media_abbreviations(text_lower)
        
        # 3. Check for emojis
        emoji_info = self.extract_emojis(text)
        
        # 4. Check for hashtags
        hashtag_info = self.extract_hashtags(text_lower)
        
        # Determine what type of error occurred (e.g., false positive, false negative)
        error_type = None
        if (original_prediction == 0 and corrected_prediction == 2) or (original_prediction == 2 and corrected_prediction == 0):
            error_type = "polarity_error"  # Complete polarity reversal
        elif abs(original_prediction - corrected_prediction) == 1:
            error_type = "intensity_error"  # Intensity error (e.g., positive vs. very positive)
        elif original_prediction == 1 and corrected_prediction != 1:
            error_type = "neutrality_error"  # Incorrectly classified as neutral
        elif original_prediction != 1 and corrected_prediction == 1:
            error_type = "sentiment_error"  # Should have been neutral
        
        # Create feedback entry
        feedback_entry = {
            "text": text,
            "original_prediction": original_prediction,
            "corrected_prediction": corrected_prediction,
            "error_type": error_type,
            "has_negation": negated,
            "has_slang": slang_info["has_abbreviations"],
            "slang_terms": list(slang_info["abbreviations"].keys()) if slang_info["has_abbreviations"] else [],
            "has_emojis": len(emoji_info["emojis"]) > 0,
            "emojis": emoji_info["emojis"] if emoji_info["emojis"] else [],
            "has_hashtags": len(hashtag_info["hashtags"]) > 0,
            "hashtags": hashtag_info["hashtags"] if hashtag_info["hashtags"] else [],
            "timestamp": str(datetime.datetime.now())
        }
        
        # Save feedback for future model improvements
        if save_feedback:
            feedback_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "feedback")
            os.makedirs(feedback_dir, exist_ok=True)
            feedback_file = os.path.join(feedback_dir, "sentiment_feedback.jsonl")
            
            with open(feedback_file, "a") as f:
                f.write(json.dumps(feedback_entry) + "\n")
        
        # Generate improvement suggestions
        suggestions = []
        
        if error_type == "polarity_error":
            if negated:
                suggestions.append("Consider strengthening negation handling for this type of text")
            if slang_info["has_abbreviations"]:
                suggestions.append(f"Update slang dictionary for terms: {', '.join(slang_info['abbreviations'].keys())}")
            if emoji_info["emojis"]:
                suggestions.append("Review emoji sentiment assignments")
        elif error_type == "neutrality_error":
            suggestions.append("Consider adjusting neutral threshold for similar content")
            
        # Update the internal model state (simple version - just add to cache)
        # For a robust implementation, this would involve online learning or model parameter adjustments
        if text not in self.analysis_cache:
            self.analysis_cache[text] = {}
        self.analysis_cache[text]["corrected_sentiment"] = corrected_prediction
        
        return {
            "success": True,
            "feedback_saved": save_feedback,
            "error_type": error_type,
            "suggestions": suggestions,
            "feedback_entry": feedback_entry
        }

    def detect_specific_emotions(self, text):
        """
        Detect specific emotions in text beyond just positive/negative sentiment.
        This helps match appropriate emojis to the detected emotions.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict with detected emotions and their confidence scores
        """
        # Define emotion keywords dictionary
        emotion_keywords = {
            "joy": [
                "happy", "joyful", "delighted", "thrilled", "excited", "ecstatic", "glad", 
                "pleased", "content", "satisfied", "cheerful", "merry", "jolly", "enjoy",
                "enjoying", "enjoyed", "fun", "wonderful", "fantastic", "awesome", "amazing",
                "celebrate", "celebrating", "celebration", "congratulations", "congrats",
                "love", "lovely", "delightful", "joy", "yay", "hooray", "woohoo", "happiness"
            ],
            "sadness": [
                "sad", "unhappy", "depressed", "depressing", "miserable", "heartbroken", "down",
                "blue", "gloomy", "somber", "melancholy", "grief", "grieving", "sorrow", "sorry",
                "regret", "regretful", "disappointed", "disappointing", "upset", "crying", "cry",
                "tears", "teary", "weeping", "weep", "mourning", "mourn", "hurt", "pain", "painful",
                "devastated", "devastation", "despair", "despairing", "hopeless", "heartache"
            ],
            "anger": [
                "angry", "mad", "furious", "enraged", "outraged", "irate", "livid", "annoyed",
                "irritated", "frustrated", "frustrating", "infuriated", "pissed", "fuming", "rage",
                "raging", "hatred", "hate", "resent", "resentment", "disgusted", "disgusting",
                "appalled", "appalling", "revolting", "revulsion", "offended", "offensive", "hostile"
            ],
            "fear": [
                "afraid", "scared", "frightened", "terrified", "fear", "fearful", "horrified", 
                "horror", "panic", "panicked", "anxious", "anxiety", "nervous", "worried", "worry",
                "dread", "dreading", "alarmed", "alarming", "threatened", "threatening", "threat",
                "intimidated", "intimidating", "petrified", "terror", "uneasy", "apprehensive"
            ],
            "surprise": [
                "surprised", "shocking", "shocked", "amazed", "astonished", "astounded", "stunned",
                "startled", "unexpected", "wow", "whoa", "omg", "oh my god", "oh my gosh", "gosh",
                "unbelievable", "incredible", "remarkable", "extraordinary", "mind-blowing", "speechless"
            ],
            "disgust": [
                "disgusted", "revolted", "repulsed", "gross", "nasty", "yuck", "ew", "eww", "ugh",
                "sickening", "sickened", "nauseous", "nauseating", "repulsive", "repugnant", "distasteful",
                "offensive", "foul", "vile", "filthy", "loathsome"
            ],
            "appreciation": [
                "thank", "thanks", "thankful", "grateful", "gratitude", "appreciate", "appreciative",
                "blessed", "fortunate", "lucky", "honored", "moved", "touching", "touched"
            ],
            "amusement": [
                "funny", "hilarious", "lol", "rofl", "lmao", "lmfao", "haha", "hehe", "amusing",
                "amused", "laughing", "laugh", "humorous", "humor", "joke", "joking", "witty",
                "comical", "entertaining", "entertained", "giggle", "giggling", "chuckle"
            ],
            "excitement": [
                "excited", "thrilled", "eager", "enthusiastic", "pumped", "stoked", "psyched",
                "hyped", "cant wait", "can't wait", "looking forward", "anticipation", "anticipating"
            ],
            "love": [
                "love", "adore", "cherish", "affection", "affectionate", "fond", "devoted", "smitten",
                "infatuated", "enamored", "romantic", "passion", "passionate", "heart", "hearts",
                "beloved", "darling", "dear", "sweetheart", "sweetie", "honey", "xoxo"
            ]
        }
        
        # Initialize emotion scores
        emotion_scores = {emotion: 0 for emotion in emotion_keywords}
        text_lower = text.lower()
        
        # Check for each emotion's keywords in the text
        for emotion, keywords in emotion_keywords.items():
            # Count occurrences of each keyword
            matches = [keyword for keyword in keywords if keyword in text_lower]
            # Score is normalized by the number of keywords in the category to avoid bias
            if matches:
                # Basic score based on number of matches
                emotion_scores[emotion] = len(matches) / len(keywords)
                
                # Boost score if keywords appear with intensifiers
                for match in matches:
                    for intensifier in self.intensifiers:
                        if f"{intensifier} {match}" in text_lower:
                            emotion_scores[emotion] += 0.5
                            break
        
        # Check for emojis that represent specific emotions
        emoji_info = self.extract_emojis(text)
        
        # Map emojis to emotions
        emoji_emotion_map = {
            "😀": "joy", "😃": "joy", "😄": "joy", "😁": "joy", "😆": "joy", "😅": "joy", "🤣": "joy",
            "😂": "joy", "🙂": "joy", "😊": "joy", "😍": "love", "🥰": "love", "😘": "love",
            "😢": "sadness", "😭": "sadness", "😞": "sadness", "😔": "sadness", "😟": "sadness",
            "😠": "anger", "😡": "anger", "🤬": "anger", "😤": "anger",
            "😨": "fear", "😱": "fear", "😰": "fear", "😥": "fear",
            "😲": "surprise", "😯": "surprise", "😮": "surprise", "😦": "surprise",
            "🤢": "disgust", "🤮": "disgust", "😖": "disgust",
            "😌": "appreciation", "🙏": "appreciation", "🥹": "appreciation",
            "😏": "amusement", "😜": "amusement", "😝": "amusement", "😛": "amusement",
            "🤩": "excitement", "✨": "excitement", "🎉": "excitement", "🎊": "excitement"
        }
        
        # Add emotion scores based on emojis
        for emoji in emoji_info["emojis"]:
            normalized_emoji = self._normalize_emoji(emoji)
            if normalized_emoji in emoji_emotion_map:
                emotion = emoji_emotion_map[normalized_emoji]
                emotion_scores[emotion] += 1
        
        # Determine primary and secondary emotions
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out emotions with zero score
        valid_emotions = [emotion for emotion, score in sorted_emotions if score > 0]
        
        # Get the top two emotions
        primary_emotion = sorted_emotions[0][0] if sorted_emotions and sorted_emotions[0][1] > 0 else None
        secondary_emotion = sorted_emotions[1][0] if len(sorted_emotions) > 1 and sorted_emotions[1][1] > 0 else None
        
        # Calculate confidence for primary emotion (normalized to 0-100%)
        primary_confidence = 0
        if primary_emotion:
            # Base confidence on the score relative to the sum of all scores
            total_score = sum(score for _, score in sorted_emotions if score > 0)
            if total_score > 0:
                primary_confidence = int(min(100, (sorted_emotions[0][1] / total_score) * 100))
        
        result = {
            "primary_emotion": primary_emotion,
            "secondary_emotion": secondary_emotion,
            "primary_confidence": primary_confidence,
            "all_emotions": {emotion: score for emotion, score in sorted_emotions if score > 0},
            "valid_emotions": valid_emotions
        }
        
        return result

    def analyze_sentiment(self, text):
        """Analyze the sentiment of the given text using the configured model and enhancements.
        
        Args:
            text: The text to analyze
            
        Returns:
            A dict with sentiment analysis results including label and score
        """
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid input text: {text}")
            return {"label": "NEUTRAL", "score": 0.5}
            
        # Log the request with timestamp
        log_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"[{log_timestamp}] Analyzing sentiment for text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # Summarize long texts if BART is enabled
        original_text = text
        if self.use_bart and len(text.split()) > 100:
            try:
                summary = self.bart_pipeline(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                logger.info(f"Summarized text using BART from {len(text.split())} words to {len(summary.split())} words")
                text = summary
            except Exception as e:
                logger.error(f"Error during BART summarization: {str(e)}")
        
        # Enhance with RAG if enabled
        enhanced_text = text
        if self.use_rag and self.rag:
            try:
                context = self.rag.retrieve_context(text)
                enhanced_text = self.rag.enhance_input(text, context)
                logger.info(f"Enhanced text with RAG context")
            except Exception as e:
                logger.error(f"Error during RAG enhancement: {str(e)}")
                
        # Analyze sentiment using the pre-trained model
        try:
            # Use the enhanced text for sentiment analysis
            result = self.analyzer(enhanced_text)[0]
            sentiment = result["label"]
            confidence = result["score"]
            
            # RoBERTa model uses different labels, need to standardize
            if self.model_type == "roberta":
                # Map RoBERTa labels to our standard format
                label_mapping = {
                    "LABEL_0": "NEGATIVE",
                    "LABEL_1": "NEUTRAL",
                    "LABEL_2": "POSITIVE"
                }
                sentiment = label_mapping.get(sentiment, sentiment)
            
            # Log the results
            logger.info(f"Sentiment analysis result: {sentiment} with confidence {confidence:.4f}")
            
            # Add metadata about RAG and BART usage
            result["enhanced_with_rag"] = self.use_rag and self.rag is not None
            result["summarized_with_bart"] = self.use_bart and len(original_text.split()) > 100
            
            return result
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {str(e)}")
            return {"label": "NEUTRAL", "score": 0.5}

def create_and_test_model():
    """Create a sentiment analysis model and test it on sample texts."""
    model = SentimentModel()
    
    # Test the model on diverse sample texts
    print("\nTesting model predictions:")
    test_texts = [
        "This is absolutely amazing! Best experience of my life!",
        "I had a really good time and enjoyed the experience.",
        "It was okay, nothing special.",
        "I didn't really enjoy this and wouldn't recommend it.",
        "Terrible experience! Complete waste of money and time!"
    ]
    
    predictions = model.predict(test_texts)
    
    # Additional tests to calibrate the sentiment thresholds
    print("\nAdditional sentiment tests:")
    calibration_texts = [
        "The absolute best product I've ever used, incredible results!",
        "I'm very satisfied with my purchase, it works well.",
        "It's an average product, does what you'd expect.",
        "The product is somewhat disappointing and below expectations.",
        "Absolutely horrible! Don't waste your money on this garbage!"
    ]
    
    calibration_results = model.predict(calibration_texts)
    
    # Extended test cases with more diverse expressions
    print("\nExtended test cases:")
    extended_test_texts = [
        # Overwhelmingly negative examples
        "This is the worst thing I have ever experienced. Absolutely terrible!",
        "I am disgusted with this service. Never coming back again!",
        "Horrible, horrible, horrible! Avoid at all costs!",
        
        # Negative examples
        "I was disappointed with how this turned out.",
        "There were several issues that made this experience unpleasant.",
        "Not what I expected, and I feel let down.",
        
        # Neutral examples
        "It's neither good nor bad, just average.",
        "The performance was acceptable, but nothing memorable.",
        "It works as expected, nothing more, nothing less.",
        "I have mixed feelings about this product.",
        
        # Positive examples
        "I enjoyed using this product, it made my work easier.",
        "The service was good and the staff was helpful.",
        "Overall a pleasant experience that I would recommend.",
        
        # Overwhelmingly positive examples
        "Absolutely phenomenal! This exceeded all my expectations!",
        "I'm thrilled with the results! Best decision I've ever made!",
        "Outstanding quality and service! 10/10 would recommend!"
    ]
    
    print("\nExtended test results:")
    extended_results = model.predict(extended_test_texts)
    
    # Edge cases to test boundary conditions
    print("\nEdge cases:")
    edge_cases = [
        # Short texts
        "Good.",
        "Bad.",
        "Okay.",
        "Meh.",
        
        # Mixed sentiment
        "While I loved the service, the product itself was disappointing.",
        "Despite some flaws, I was generally satisfied with my purchase.",
        "Amazing features but terrible customer support.",
        
        # Neutral with subtle sentiment
        "It's just fine I guess, if you like that sort of thing.",
        "Well, it could be worse, but also could be much better."
    ]
    
    print("\nEdge case results:")
    edge_case_results = model.predict(edge_cases)
    
    # Additional test cases for hashtag analysis
    print("\nTesting hashtag analysis:")
    hashtag_test_texts = [
        "What a game! #winning #blessed #happy",
        "Another disappointing loss #sad #frustrated #failure",
        "Mixed feelings about this new policy #hopeful but also #worried",
        "Beautiful day at the beach #relaxed #peaceful #joy",
        "Can't believe how bad this service was #awful #waste #disappointed",
        "Just a regular day #monday #working",
        "Leonardo DiCaprio still partying with 19-year-olds #eyeroll #cringe #yikes",
        "New research shows promising results #science #breakthrough #excited"
    ]
    
    hashtag_results = model.predict(hashtag_test_texts)
    
    # Calculate statistics on test results
    all_predictions = predictions + calibration_results + extended_results + edge_case_results + hashtag_results
    count_by_category = {i: all_predictions.count(i) for i in range(3)}
    print("\nDistribution of predictions:")
    for category_index, count in count_by_category.items():
        category_name = model.get_sentiment_label(category_index)
        percentage = (count / len(all_predictions)) * 100
        print(f"{category_index} ({category_name}): {count} ({percentage:.1f}%)")
    
    # Save the model
    backend_path = "backend/model.pkl"
    joblib.dump(model, backend_path)
    print(f"Model saved to {backend_path}")


if __name__ == "__main__":
    create_and_test_model()

# Define the detect_hateful_content function
def detect_hateful_content(text):
    hateful_keywords = ["racist", "sexist", "hateful"]
    for keyword in hateful_keywords:
        if keyword in text.lower():
            return True
    return False

# Define the analyze_sentiment function
def analyze_sentiment(text):
    try:
        if detect_hateful_content(text):
            return {"label": "HATEFUL", "score": 1.0}

        # Placeholder for actual sentiment analysis logic
        result = {"label": "NEUTRAL", "score": 0.5}
        logger.info(f"Sentiment analysis completed for text: {text}")
        return result
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
        raise