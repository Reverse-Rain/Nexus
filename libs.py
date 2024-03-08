import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import math
import spacy
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import FastText
from laserembeddings import Laser
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet
