import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from fastapi import FastAPI
import tensorflow as tf
from tensorflow import Tensor
import keras
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pydantic import BaseModel
from typing import TypeVar, Generic

T = TypeVar('T')

class ApiResponse(BaseModel, Generic[T]):
    body: T | None = None
    success: bool
    error: str | None

class ToxicContent(BaseModel):
    sentence: str
    sentence_prediction_slices: list[tuple[float, str]]
    is_toxic: bool
    raw_predictions: list[float]

app = FastAPI()
model: Sequential

@app.get("/version", response_model=ApiResponse[str])
def read_version() -> ApiResponse[str]:
    version: str = f"tensorflow: {tf.__version__}\n"
    f"keras : {keras.__version__}\n"
    return ApiResponse(body=version, success=True, error=None)

@app.post("/toxic", response_model=ApiResponse[ToxicContent])
def read_item(input: str) -> ApiResponse[ToxicContent]:
    try:
        sentences: list[str]
        predictions: list[float]
        sentences, predictions = predict(input)
        return ApiResponse(
            body=ToxicContent(
                sentence=input,
                sentence_prediction_slices=[(round(predictions[i], 4) * 100, sentence) for i, sentence in enumerate(sentences)],
                is_toxic=np.max(predictions) > 0.5,
                raw_predictions = predictions
            ),
            success=True,
            error=None
        )
    except Exception as e:
        return ApiResponse(body=None, success=False, error=f"{e}")


def predict(input: str) -> tuple[list[str], list[float]]:
    model: Sequential = keras.models.load_model('lstm_toxic_classifier_from_scratch.keras') # type: ignore
    MAX_LEN: int = 40
    SLICE_LENGTH = MAX_LEN // 4
    input_length: int = len(input)
    sentences: list[str]
    if input_length <= MAX_LEN:
        sentences = [input]
    else:
        slices_count: int = input_length // SLICE_LENGTH
        sentences = [input[i*SLICE_LENGTH:i*SLICE_LENGTH+MAX_LEN] for i in range(0, slices_count)]

    predictions: Tensor = model.predict(tf.convert_to_tensor(sentences))
    return sentences, np.array(predictions).flatten().tolist()

@keras.saving.register_keras_serializable()
def custom_standardisation(t: Tensor) -> Tensor:
    # https://github.com/google/re2/wiki/Syntax
    # Remplacement manuel des lettres accentuées
    t = tf.strings.regex_replace(t, "[éèêë]", "e")
    t = tf.strings.regex_replace(t, "[ÉÈÊË]", "E")
    t = tf.strings.regex_replace(t, "[àâä]", "a")
    t = tf.strings.regex_replace(t, "[ÀÂÄ]", "A")
    t = tf.strings.regex_replace(t, "[îï]", "i")
    t = tf.strings.regex_replace(t, "[ÎÏ]", "I")
    t = tf.strings.regex_replace(t, "[ôö]", "o")
    t = tf.strings.regex_replace(t, "[ÔÖ]", "O")
    t = tf.strings.regex_replace(t, "[ùûü]", "u")
    t = tf.strings.regex_replace(t, "[ÙÛÜ]", "U")
    t = tf.strings.regex_replace(t, "ç", "c")
    t = tf.strings.regex_replace(t, "Ç", "C")
    t = tf.strings.regex_replace(t, "ÿ", "y")
    t = tf.strings.regex_replace(t, "Ÿ", "Y")

    t = tf.strings.lower(t)
    t = tf.strings.regex_replace(t, r"[a-z]+://[^ ]+", "") # retrait d'url
    t = tf.strings.regex_replace(t, r"<[^>]+>", " ") # strip html tags

    t = tf.strings.regex_replace(t, r"\pP", "") # Retrait de poncutation
    t = tf.strings.regex_replace(t, r"\pS", "") # Retrait de Symboles
    t = tf.strings.regex_replace(t, r"[^\p{Latin} ]", "") # Retrait de poncutation
    
    t = tf.strings.regex_replace(t, r"(\b\w\b)", "") # retrait des mots de moins de 2 lettres
    t = tf.strings.regex_replace(t, r"\s+", " ") # retrait des espaces en trop

    t = tf.strings.strip(t)

    t = tf.strings.regex_replace(t, r"^$", "[UNK]") # en cas de chaine vide, renvois le tag [UNK] utilisé par défaut

    return t

