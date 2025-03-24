import nltk
import string
import numpy as np
import spacy
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
import sys
import torch

## download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('wordnet_ic', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class EpsilonDistributor:
    """
    A class for calculating epsilon distribution for a given sentence.
    """
    
    def __init__(self, model_checkpoint="thenlper/gte-small"):
        """
        Initializes the EpsilonDistributor class.
        """
        if torch.cuda.is_available() == True:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.wordnet_ic = {
            'semcor': wordnet_ic.ic('ic-semcor.dat'),
            'brown': wordnet_ic.ic('ic-brown.dat'),
            'bnc': wordnet_ic.ic('ic-bnc.dat'),
            'shaks':wordnet_ic.ic('ic-shaks.dat'),
            'treebank': wordnet_ic.ic('ic-treebank.dat')
        }

        self.model = SentenceTransformer(model_checkpoint, device=self.device)
        self.nlp = spacy.load("en_core_web_md")
        self.pos_informativeness = {'NN': 14, 'PR':7, 'VB':15, 'CD':2, 'JJ':5, 'RB':5}
        self.lemmatizer = WordNetLemmatizer()


    def _go_through_wordnet(self, ss):
        """
        Helper function for _get_ic. Calculates the information content (IC) values for synsets in WordNet.

        Args:
            ss (list): List of synsets.

        Returns:
            list: List of IC values.
        """
        ic_values = []
        for corpus_ic in self.wordnet_ic.values():
                max_ic_values = [corpus_ic['n'].get(x.offset(), 0) for x in ss]
                if max_ic_values:  # Check if the list is not empty
                    max_ic = np.mean(max_ic_values)
                    ic_values.append(max_ic)

        return ic_values

    def _get_ic(self, tokens):
        """
        Retrieves the information content (IC) values for tokens in the sentence.

        Returns:
            list: List of tuples containing token and its IC value.
        """
        words_ics = []

        for t in tokens:
            tt = t.lower()

            lema_tt = self.lemmatizer.lemmatize(tt)
            v_tt = self.lemmatizer.lemmatize(tt, "v")
            n_tt = self.lemmatizer.lemmatize(tt, "n")

            if lema_tt == v_tt and lema_tt == n_tt:
                tt = lema_tt
            elif lema_tt != v_tt and lema_tt == n_tt:
                tt = v_tt
            else:
                tt = n_tt
            ss = wn.synsets(tt)
            ic_values = self._go_through_wordnet(ss)
            ic = 1.0 if (not ic_values) else sum(ic_values) / len(ic_values)
            if ic == 0:
                ic = 1.0
            words_ics.append((t, ic))        
        return words_ics


    def _get_pos_informativeness(self, tokens):
        """
        Calculates the part-of-speech (POS) informativeness scores for tokens in the sentence based on the weight.

        Returns:
            list: List of tuples containing token and its POS informativeness score.
        """
        pos_tags = pos_tag(tokens)
        word_informativeness = [(word, self.pos_informativeness.get(pos[:2], 0.1) if pos not in self.pos_informativeness else self.pos_informativeness.get(pos, 0.1))
                            for word, pos in pos_tags]
        return word_informativeness

    def set_pos_weights(self, weights):
        """
        Sets custom weights for part-of-speech (POS) informativeness scores.

        Args:
            weights (dict): Dictionary containing custom weights for POS tags.
        """
        if weights == {}:
            self.pos_informativeness = self.pos_informativeness
        else:
            self.pos_informativeness = weights

    def _get_ner_weights(self, sentence):
        """
        Calculates the named entity recognition (NER) weights for tokens in the sentence.

        Returns:
            list: List of tuples containing token and its NER weight.
        """
        doc = self.nlp(sentence)
        word_weights = [(word.text, 1) if word.ent_type_ else (word.text, 0) for word in doc]
        return word_weights

    def _get_similarity_except_word_score(self, tokens):
        """
        Calculates the similarity scores for the sentence with each word removed.

        Returns:
            list: List of tuples containing token and its similarity score.
        """
        word_removed_measure = []
        sentences_without_word = []   
        for idx, t in enumerate(tokens):
            sentences_without_word.append(' '.join([w for i, w in enumerate(tokens) if i != idx]))
        sentences_without_word_embedding = self.model.encode(sentences_without_word)   

        for word, sentence_without_word_embedding in zip(tokens, sentences_without_word_embedding):
            similarity = np.dot(self.original_embedding, sentence_without_word_embedding) / (np.linalg.norm(self.original_embedding) * np.linalg.norm(sentence_without_word_embedding))
            importance_measure = abs(1 - similarity)
            word_removed_measure.append((word, importance_measure))

        return word_removed_measure

    def _get_similarity_single_word_score(self, tokens):
        """
        Calculates the similarity scores for each word in the sentence.

        Returns:
            list: List of tuples containing token and its similarity score.
        """
        word_sent_sim_measure = []

        single_word_embeddings = self.model.encode(tokens)
        for word, single_word_embedding in zip(tokens, single_word_embeddings):
            similarity = np.dot(self.original_embedding, single_word_embedding) / (np.linalg.norm(self.original_embedding) * np.linalg.norm(single_word_embedding))
            importance_measure = abs(similarity)
            word_sent_sim_measure.append((word, importance_measure))

        return word_sent_sim_measure

    def _normalize_by_total(self, tuple_list):
        """
        Normalizes the scores in a list of tuples by their total sum.

        Args:
            tuple_list (list): List of tuples containing token and its score.

        Returns:
            list: List of tuples containing token and its normalized score.
        """
        total_sum = sum(score for _, score in tuple_list)
        normalized_scores = [(word, score / total_sum) for word, score in tuple_list]
        return normalized_scores

    def _combine_scores(self, tokens, sentence, total_epsilon, use_ner=True, use_ic=True, use_pos=True, use_sim_sent=True, use_sim_word=True):
        """
        Combines different scores based on specified parameters and calculates epsilon distribution.

        Args:
            total_epsilon (float): Total epsilon value for distribution.
            use_ner (bool): Flag to indicate whether to use named entity recognition (NER) scores.
            use_ic (bool): Flag to indicate whether to use information content (IC) scores.
            use_pos (bool): Flag to indicate whether to use part-of-speech (POS) scores.
            use_sim_sent (bool): Flag to indicate whether to use similarity scores with sentence.
            use_sim_word (bool): Flag to indicate whether to use similarity scores with individual words.

        Returns:
            tuple: Tuple containing final scores, epsilon proportion, and distributed epsilon.
        """
        # Initialize default scores for words in the sentence
        n = len(tokens)
        default_scores = [(word, 0) for word in tokens]
        # Initialize lists to store scores and flags for enabled scores
        score_count = 0

        if use_ic :
            ic_scores = self._get_ic(tokens)
            normalized_ic_scores = self._normalize_by_total(ic_scores)
            score_count += 1
        else:
            normalized_ic_scores = default_scores
        if use_pos:
            pos_scores = self._get_pos_informativeness(tokens)
            normalized_pos_scores = self._normalize_by_total(pos_scores)
            score_count += 1
        else:
            normalized_pos_scores = default_scores

        if use_sim_sent:
            sim_sent_scores = self._get_similarity_except_word_score(tokens)
            normalized_sim_sent_scores = self._normalize_by_total(sim_sent_scores)
            score_count += 1
        else:
            normalized_sim_sent_scores = default_scores

        if use_sim_word:
            sim_word_scores = self._get_similarity_single_word_score(tokens)
            normalized_sim_word_scores = self._normalize_by_total(sim_word_scores)
            score_count += 1
        else:
            normalized_sim_word_scores = default_scores

        if use_ner:
            ner_scores = self._get_ner_weights(sentence)
            if 1 in ner_scores:
                normalized_ner_scores = self._normalize_by_total(ner_scores)
                score_count += 1
            else:
                normalized_ner_scores = default_scores
        else:
            normalized_ner_scores = default_scores

        combined_scores = []
        total_score = 0.0

        # Combine scores
        for idx in range(n):
            token = normalized_ic_scores[idx][0]
            ic = normalized_ic_scores[idx][1]
            pos = normalized_pos_scores[idx][1]
            sent_sim = normalized_sim_sent_scores[idx][1]
            word_sim = normalized_sim_word_scores[idx][1]
            ner = normalized_ner_scores[idx][1]
            combined_score = (ic + pos + sent_sim + word_sim + ner) / score_count
            combined_scores.append((token, combined_score))
            total_score += combined_score

        # Normalize scores
        scale_factor = 1 / total_score if total_score != 0 else 0
        normalized_scores = [(word, score * scale_factor) for word, score in combined_scores]

        # Calculate epsilon proportion and distribute epsilon
        closest_positive_to_zero = sys.float_info.min
        inversed_scores = [ (word, (1 / (score * 100))) if score != 0 else (1 / (closest_positive_to_zero * 100)) for word, score in normalized_scores]
        inversed_scores_sum = sum(score for _, score in inversed_scores)
        ep_propotion = [ (word, ((1 / (score * 100)) / inversed_scores_sum) if score != 0 else ((1 / (closest_positive_to_zero * 100)) / inversed_scores_sum)) for word, score in normalized_scores]
        distributed_epsilon = [ (word, total_epsilon * score) for word, score in ep_propotion]

        return normalized_scores, ep_propotion, distributed_epsilon

    def get_distribution(self, sentence, total_epsilon=100, use_ner=True, use_ic=True, use_pos=True, use_sim_sent=True, use_sim_word=True):
        """
        Calculates the distribution of epsilon values for each token in the given sentence based on various scoring
        criteria.

        Args:
            sentence (str): The input sentence for which epsilon distribution is to be calculated.
            total_epsilon (float): Total epsilon value for distribution.
            use_ner (bool): Flag to indicate whether to use named entity recognition (NER) scores.
            use_ic (bool): Flag to indicate whether to use information content (IC) scores.
            use_pos (bool): Flag to indicate whether to use part-of-speech (POS) scores.
            use_sim_sent (bool): Flag to indicate whether to use similarity scores with sentence.
            use_sim_word (bool): Flag to indicate whether to use similarity scores with individual words.

        Returns:
            list: List of tuples containing token and its distributed epsilon value.
        """
        
        tokens = nltk.word_tokenize(sentence)
        punct = [(i, x) for i, x in enumerate(tokens) if x.isalnum() == False]
        tokens = [x for i, x in enumerate(tokens) if (i, x) not in punct]
      
        if use_sim_sent or use_sim_word:
            self.original_embedding = self.model.encode([sentence])[0]

        final_scores, final_norm_epsilon, final_distributed_epsilon = self._combine_scores(tokens=tokens, sentence=sentence, total_epsilon=total_epsilon, use_ner=use_ner, use_ic=use_ic, use_pos=use_pos, use_sim_sent=use_sim_sent, use_sim_word=use_sim_word)

        for tup in punct:
            final_distributed_epsilon.insert(tup[0], (tup[1], None))

        return final_distributed_epsilon