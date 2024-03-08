nltk.download('punkt')
nltk.download('stopwords')

class SimilarityModels:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.model_paraphrase = SentenceTransformer('paraphrase-distilroberta-base-v1')
        self.module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.use_model = hub.load(self.module_url)
        self.model_sbert = SentenceTransformer('paraphrase-mpnet-base-v2')
        self.model_distilbert = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.laser_model = Laser()

    def preprocess_text(self, sentence):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(sentence.lower())
        filtered_sentence = [word for word in word_tokens if word not in stop_words]
        return filtered_sentence

    def get_cosine_similarity(self, sentence1, sentence2):
        sentence1 = self.preprocess_text(sentence1)
        sentence2 = self.preprocess_text(sentence2)

        vector1 = Counter(sentence1)
        vector2 = Counter(sentence2)

        all_words = set(vector1.keys()).union(set(vector2.keys()))

        dot_product = sum(vector1.get(word, 0) * vector2.get(word, 0) for word in all_words)

        magnitude1 = math.sqrt(sum(vector1.get(word, 0)**2 for word in all_words))
        magnitude2 = math.sqrt(sum(vector2.get(word, 0)**2 for word in all_words))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        else:
            return dot_product / (magnitude1 * magnitude2)

    def jaccard_similarity(self, sentence1, sentence2):
        stop_words = set(stopwords.words('english'))

        words1 = [word.lower() for word in word_tokenize(sentence1) if word.isalnum() and word.lower() not in stop_words]
        words2 = [word.lower() for word in word_tokenize(sentence2) if word.isalnum() and word.lower() not in stop_words]

        intersection = len(set(words1).intersection(set(words2)))
        union = len(set(words1).union(set(words2)))

        if union == 0:
            return 0
        else:
            return intersection / union

    def word_movers_distance(self, sentence1, sentence2):
        doc1 = self.nlp(sentence1)
        doc2 = self.nlp(sentence2)

        wmd = doc1.similarity(doc2)

        return wmd

    def sentence_similarity_transformers(self, sentence1, sentence2):
        embeddings1 = self.model_paraphrase.encode(sentence1, convert_to_tensor=True)
        embeddings2 = self.model_paraphrase.encode(sentence2, convert_to_tensor=True)

        similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2)

        return similarity_score.item()

    def sentence_similarity_USE(self, sentence1, sentence2):
        embeddings = self.use_model([sentence1, sentence2])

        similarity_score = tf.keras.losses.cosine_similarity(embeddings[0], embeddings[1]).numpy()

        return similarity_score

    def create_doc2vec_model(self, sentences):
        documents = [TaggedDocument(self.preprocess_text(sentence), [i]) for i, sentence in enumerate(sentences)]
        model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)
        return model

    def sentence_similarity_doc2vec(self, sentence1, sentence2, model):
        vec1 = model.infer_vector(self.preprocess_text(sentence1))
        vec2 = model.infer_vector(self.preprocess_text(sentence2))

        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        return similarity

    def create_fasttext_model(self, sentences):
        model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4)
        return model

    def sentence_similarity_fasttext(self, sentence1, sentence2, model):
        vec1 = model.wv[sentence1]
        vec2 = model.wv[sentence2]

        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        return similarity

    def sentence_similarity_sbert(self, sentence1, sentence2):
        embeddings1 = self.model_sbert.encode([sentence1])
        embeddings2 = self.model_sbert.encode([sentence2])

        similarity = np.dot(embeddings1[0], embeddings2[0]) / (np.linalg.norm(embeddings1[0]) * np.linalg.norm(embeddings2[0]))

        return similarity

    def sentence_similarity_distilbert(self, sentence1, sentence2):
        embeddings1 = self.model_distilbert.encode([sentence1])
        embeddings2 = self.model_distilbert.encode([sentence2])

        similarity = np.dot(embeddings1[0], embeddings2[0]) / (np.linalg.norm(embeddings1[0]) * np.linalg.norm(embeddings2[0]))

        return similarity

    def sentence_similarity_laser(self, sentence1, sentence2):
        embeddings1 = self.laser_model.embed_sentences([sentence1], lang='en')
        embeddings2 = self.laser_model.embed_sentences([sentence2], lang='en')

        similarity = np.dot(embeddings1[0], embeddings2[0]) / (np.linalg.norm(embeddings1[0]) * np.linalg.norm(embeddings2[0]))

        return similarity
            
# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def calculate_sentence_similarity(reference_sentence, student_answer):
    def get_synonyms_antonyms(word):
        synonyms = []
        antonyms = []

        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
                if lemma.antonyms():
                    antonyms.append(lemma.antonyms()[0].name())

        return synonyms, antonyms

    def get_synonyms_antonyms_for_sentence(sentence):
        words = nltk.word_tokenize(sentence)

        synonyms_list = []
        antonyms_list = []

        for word in words:
            synonyms, antonyms = get_synonyms_antonyms(word)
            synonyms_list.append(synonyms)
            antonyms_list.append(antonyms)

        result = {
            "sentence": sentence,
            "words": words,
            "synonyms": dict(zip(words, synonyms_list)),
            "antonyms": dict(zip(words, antonyms_list))
        }

        return result

    result1 = get_synonyms_antonyms_for_sentence(reference_sentence)
    result2 = get_synonyms_antonyms_for_sentence(student_answer)

    synonyms1 = result1["synonyms"]
    synonyms2 = result2["synonyms"]

    antonyms1 = result1["antonyms"]
    antonyms2 = result2["antonyms"]

    synonyms1_set = set([syn for synonyms in synonyms1.values() for syn in synonyms])
    synonyms2_set = set([syn for synonyms in synonyms2.values() for syn in synonyms])

    antonyms1_set = set([ant for antonyms in antonyms1.values() for ant in antonyms])
    antonyms2_set = set([ant for antonyms in antonyms2.values() for ant in antonyms])

    set1 = synonyms1_set.union(antonyms1_set)
    set2 = synonyms2_set.union(antonyms2_set)

    doc1 = nlp(" ".join(set1))
    doc2 = nlp(" ".join(set2))

    similarity_score = doc1.similarity(doc2)

    return similarity_score


