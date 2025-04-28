import joblib
import torch
from transformers import pipeline
import numpy as np
import re

class SentimentModel:
    """A wrapper around Hugging Face's sentiment analysis pipeline for nuanced sentiment analysis."""
    
    def __init__(self):
        """Initialize the sentiment analyzer using a pre-trained model."""
        # Use a model specifically fine-tuned for sentiment analysis
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        
        # Try to use CUDA if available
        device = 0 if torch.cuda.is_available() else -1
        print(f"Device set to use {'cuda:' + str(device) if device >= 0 else 'CPU'}")
        
        # Create the sentiment analysis pipeline
        self.analyzer = pipeline("sentiment-analysis", model=model_name, device=device)
        
        # Define the sentiment categories
        self.sentiment_categories = [
            "overwhelmingly negative",
            "negative",
            "neutral",
            "positive",
            "overwhelmingly positive"
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
            "value", "valuable", "handy", "practical", "efficient", "well done"
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
            "masterful", "masterpiece", "couldn't be better", "absolutely loved"
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
            "not good", "not great", "not recommended", "not impressed"
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
            "never again", "beyond awful", "completely useless", "utter disappointment"
        ]
        
        # High priority negative words that should have stronger influence on sentiment
        self.priority_negative_words = [
            "trash", "garbage", "terrible", "awful", "horrible", "hate", "disgusting",
            "useless", "waste", "pathetic", "worst"
        ]
        
        # High priority positive words
        self.priority_positive_words = [
            "amazing", "excellent", "perfect", "love", "incredible", "outstanding"
        ]
        
        # Intensifiers that amplify the sentiment
        self.intensifiers = [
            "very", "extremely", "incredibly", "absolutely", "completely", "totally",
            "utterly", "thoroughly", "entirely", "fully", "really", "quite", "so",
            "too", "especially", "particularly", "exceptionally", "tremendously",
            "immensely", "hugely", "vastly", "insanely", "ridiculously", "seriously",
            "unbelievably", "amazingly", "remarkably", "decidedly", "highly",
            "profoundly", "deeply", "overwhelmingly", "purely", "super", "mega",
            "ultra", "extra", "beyond", "exceedingly", "extraordinarily"
        ]
        
        # Common positive and negative hashtags
        self.positive_hashtags = [
            "love", "happy", "blessed", "grateful", "joy", "beautiful", "amazing", 
            "awesome", "excited", "wonderful", "success", "inspiration", "goals",
            "win", "winning", "blessed", "gratitude", "positive", "positivity",
            "goodvibes", "bestday", "perfect", "smile", "motivated", "thankful",
            "fun", "hope", "peace", "celebrate", "victory", "achievement", "proud",
            "supportive", "inspiring", "favorite", "wow", "gorgeous", "excellence",
            "champion", "yay", "congrats", "congratulations", "blessings", "goodnews"
        ]
        
        self.negative_hashtags = [
            "sad", "angry", "upset", "disappointed", "fail", "failure", "hate", 
            "depressed", "depression", "anxiety", "stressed", "tired", "exhausted",
            "worried", "heartbroken", "brokenheart", "lonely", "alone", "hurt",
            "pain", "tears", "crying", "frustrated", "mad", "annoyed", "rage",
            "badday", "disaster", "tragedy", "tragic", "disgusted", "disgusting",
            "horrible", "terrible", "awful", "ugh", "fml", "smh", "wtf", "outraged",
            "shame", "shameful", "pathetic", "ridiculous", "meaningless", "waste",
            "useless", "hardship", "struggle", "suffering", "victim", "trauma"
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
        
        # Emoji sentiment lists
        self.positive_emojis = [
            "😀", "😃", "😄", "😁", "😆", "😊", "😇", "🙂", "😍", "🥰", "😘", "😗", "😙", "😚", "😋", "😜", "😝", "😛", "🤗", "🤩", "🥳", "😺", "😸", "😻", "👍", "👏", "💪", "🙌", "🎉", "✨", "💖", "❤️", "💕", "💞", "💓", "💗", "💙", "💚", "💛", "💜", "🧡", "🤍", "🤎", "💯", "✔️", "😎", "😌", "😺", "😸", "😹", "😻", "😽", "😼", "😺", "😸", "😹", "😻", "😽", "😼", "😺", "😸", "😹", "😻", "😽", "😼"
        ]
        self.negative_emojis = [
            "😞", "😔", "😟", "😕", "🙁", "☹️", "😣", "😖", "😫", "😩", "🥺", "😢", "😭", "😤", "😠", "😡", "🤬", "👎", "😾", "😿", "🙀", "😱", "😨", "😰", "😥", "😓", "😒", "😓", "😔", "😞", "😢", "😭", "😤", "😠", "😡", "🤬", "👎", "💔", "😾", "😿", "🙀", "😱", "😨", "😰", "😥", "😓", "😒", "😓", "😔", "😞", "😢", "😭", "😤", "😠", "😡", "🤬", "👎", "💔"
        ]
        self.neutral_emojis = [
            "😐", "😑", "😶", "🤔", "😬", "🤨", "😏", "😒", "😕", "😯", "😲", "😳", "😦", "😧", "😮", "😴", "🤤", "😪", "😵", "🤐", "🤢", "🤮", "😷", "🤒", "🤕", "🤑", "😲", "😯", "😮", "😬", "🤔", "😑", "😐", "😶"
        ]

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

    def extract_emojis(self, text):
        """Extract emojis from text and analyze their sentiment, handling variations and context-aware crying emoji."""
        try:
            import emoji
        except ImportError:
            return {"emojis": [], "positive_count": 0, "negative_count": 0, "neutral_count": 0, "sentiment_score": 0}
        found_emojis = emoji.emoji_list(text)
        emoji_chars = [item["emoji"] for item in found_emojis]
        normalized_emojis = [self._normalize_emoji(e) for e in emoji_chars]
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        for orig, norm in zip(emoji_chars, normalized_emojis):
            # Special handling for crying emoji
            if norm == "😭":
                if self._is_positive_crying_context(text):
                    positive_count += 1
                else:
                    negative_count += 1
            else:
                if hasattr(self, "positive_emojis") and norm in self.positive_emojis:
                    positive_count += 1
                elif hasattr(self, "negative_emojis") and norm in self.negative_emojis:
                    negative_count += 1
                elif hasattr(self, "neutral_emojis") and norm in self.neutral_emojis:
                    neutral_count += 1
        emoji_sentiment = 0
        total = positive_count + negative_count + neutral_count
        if total:
            emoji_sentiment = (positive_count - negative_count) / total
        return {
            "emojis": emoji_chars,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "sentiment_score": emoji_sentiment
        }

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
        return any(kw in text for kw in self.critical_keywords)

    def predict(self, texts):
        """
        Analyze sentiment and return predictions.
        Args:
            texts: A string or list of strings to analyze
        Returns:
            List of sentiment category indices (0-4) corresponding to the 5 sentiment levels
        """
        # Make sure input is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Get predictions from the pipeline
        results = self.analyzer(texts)
        
        # Overwhelmingly negative phrase list
        overwhelmingly_negative_phrases = [
            "worst experience ever", "absolutely terrible", "never again", "total disaster", "utter disappointment",
            "completely useless", "beyond awful", "absolutely hate", "worst thing ever", "run away from", "stay away"
        ]
        
        # Convert the pipeline output to our 5 sentiment categories
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
            
            # --- Initialize hashtag_info and emoji_info before use ---
            hashtag_info = self.extract_hashtags(text)
            emoji_info = self.extract_emojis(text)
            
            # Overwhelmingly negative logic (combined rules)
            is_overwhelmingly_negative = False
            negative_signals = 0
            # 1. Phrase match
            if any(phrase in text for phrase in overwhelmingly_negative_phrases):
                negative_signals += 1
            # 2. Very negative or priority negative keyword
            if self.detect_keywords(text, self.very_negative_keywords):
                negative_signals += 1
            if self.detect_keywords(text, self.priority_negative_words):
                negative_signals += 1
            # 3. Intensifier before negative/priority negative word
            has_intensified_negative, _ = self.check_intensified_word(text, self.priority_negative_words + self.very_negative_keywords)
            if has_intensified_negative:
                negative_signals += 1
            # 4. High negative model score
            if label == "NEGATIVE" and score > 0.88:
                negative_signals += 1
            # 5. Short text with strong negative word
            if len(text.split()) <= 7 and (self.detect_keywords(text, self.very_negative_keywords) or self.detect_keywords(text, self.priority_negative_words)):
                negative_signals += 1
            # 6. Multiple negative hashtags or emojis
            if hashtag_info["negative_hashtags"] and len(hashtag_info["negative_hashtags"]) > 1:
                negative_signals += 1
            if emoji_info["negative_count"] > 1:
                negative_signals += 1
            # Require at least 2 strong negative signals for overwhelmingly negative
            if negative_signals >= 2:
                category = 0
                predictions.append(category)
                sentiment_label = self.get_sentiment_label(category)
                print(f"'{texts[i]}' → {sentiment_label} (overwhelmingly negative rule)")
                continue
            
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
                category = 2  # neutral/mixed
                debug_info["features"].append("strong mixed signals (positive & negative)")
            elif has_mixed:
                category = 2
                debug_info["features"].append("mixed/ambiguous keywords or patterns")
            # --- Critical/constructive logic ---
            elif has_critical:
                # If critical/constructive, lean toward neutral or negative, but boost weight
                if has_negative or found_priority_negative:
                    category = 1  # negative
                else:
                    category = 2  # neutral/constructive
                debug_info["features"].append("critical/constructive keywords detected (weighted)")
            else:
                # Advanced sentiment classification rules
                if has_intensified_negative or (found_priority_negative and found_intensifiers):
                    # Strongly prioritize intensified negative words like "incredibly trash"
                    category = 0 if not has_negation else 3
                elif has_intensified_positive or (found_priority_positive and found_intensifiers):
                    # Strongly prioritize intensified positive words
                    category = 4 if not has_negation else 1
                elif found_priority_negative:
                    # Prioritize high-impact negative words even without intensifiers
                    category = 0 if len(found_priority_negative) > 1 else (1 if not has_negation else 3)
                elif found_priority_positive:
                    # Prioritize high-impact positive words even without intensifiers
                    category = 4 if len(found_priority_positive) > 1 else (3 if not has_negation else 1)
                elif has_mixed:
                    # Mixed sentiment typically indicates a neutral overall sentiment
                    category = 2
                elif has_negation:
                    if label == "POSITIVE":
                        category = 1
                    else:
                        category = 3
                elif label == "NEGATIVE":
                    negative_criteria = [
                        has_very_negative,
                        has_negative,
                        has_negative_hashtags and len(hashtag_info["negative_hashtags"]) > 0,
                        has_negative_emojis and emoji_info["negative_count"] > 0,
                        score > 0.9
                    ]
                    if label == "NEGATIVE" and sum(1 for c in negative_criteria if c) >= 2:
                        category = 1  # negative
                    if has_very_negative or (has_negative_hashtags and hashtag_sentiment < -0.5) or (has_negative_emojis and emoji_sentiment < -0.5):
                        category = 0  # overwhelmingly negative
                    elif has_negative or (has_negative_hashtags) or (has_negative_emojis) or score > 0.9:
                        category = 1  # negative
                    elif has_neutral:
                        category = 2  # neutral
                    else:
                        # Lower confidence negative statements or without strong keywords
                        category = 1  # default to negative for NEGATIVE label
                else:  # label is POSITIVE
                    if has_very_positive or (has_positive_hashtags and hashtag_sentiment > 0.5) or (has_positive_emojis and emoji_sentiment > 0.5):
                        category = 4  # overwhelmingly positive
                    elif has_positive or (has_positive_hashtags) or (has_positive_emojis) or score > 0.9:
                        category = 3  # positive
                    elif has_neutral:
                        category = 2  # neutral
                    else:
                        # Lower confidence positive statements or without strong keywords
                        category = 3  # default to positive for POSITIVE label
                
                # Check for neutral overrides - these take precedence unless we have priority words
                if not (found_priority_negative or found_priority_positive):
                    if (len(text.split()) <= 3) and has_neutral:  # Short neutral statements
                        category = 2
                    elif has_neutral and not (has_very_positive or has_very_negative):
                        if not (has_positive and score > 0.95) and not (has_negative and score > 0.95):
                            # Don't override if we have strong hashtag signals
                            if not (has_positive_hashtags and hashtag_sentiment > 0.5) and not (has_negative_hashtags and hashtag_sentiment < -0.5):
                                category = 2
                
                # Hashtag override - strong hashtag signals can override weak model predictions
                if hashtag_info["hashtags"]:
                    if hashtag_sentiment > 0.7 and category < 3:  # Strong positive hashtags
                        category = 3  # Bump to at least positive
                    elif hashtag_sentiment < -0.7 and category > 1:  # Strong negative hashtags
                        category = 1  # Bump down to at least negative
                
                # Emoji override
                if emoji_info["emojis"] and len(emoji_info["emojis"]) > 0:
                    if emoji_sentiment > 0.8 and category < 4:
                        category = 4
                    elif emoji_sentiment > 0.6 and category < 3:
                        category = 3
                    elif emoji_sentiment < -0.8 and category > 0:
                        category = 0
                    elif emoji_sentiment < -0.6 and category > 1:
                        category = 1
                
                # Sarcasm flips sentiment
                if sarcasm:
                    if label == "POSITIVE":
                        category = 1
                    else:
                        category = 3
                    predictions.append(category)
                    continue
                # Emoji sequence override
                if emoji_sequence_sentiment > 0 and category < 3:
                    category = 3
                elif emoji_sequence_sentiment < 0 and category > 1:
                    category = 1
                # Negation scope override
                if neg_scope_pos:
                    category = 1
                if neg_scope_neg:
                    category = 3
                # Intensifier adjustment
                if intensifier > 1.0:
                    if category == 3:
                        category = 4
                    elif category == 1:
                        category = 0
                # Domain-specific lexicon
                if domain_pos and category < 3:
                    category = 3
                if domain_neg and category > 1:
                    category = 1
            
            predictions.append(category)
            
            # Print debug information
            sentiment_label = self.get_sentiment_label(category)
            print(f"'{texts[i]}' → {sentiment_label}")
            print(f"  Label: {debug_info['label']}, Score: {debug_info['score']}")
            if hashtag_info["hashtags"]:
                print(f"  Hashtag sentiment: {hashtag_sentiment:.2f}")
            if debug_info["features"]:
                print(f"  Features: {', '.join(debug_info['features'])}")
        
        return predictions
    
    def get_sentiment_label(self, category_index):
        """Get the text label for a sentiment category index."""
        return self.sentiment_categories[category_index]


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
    count_by_category = {i: all_predictions.count(i) for i in range(5)}
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