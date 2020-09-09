# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:12:05 2020

@author: SaiKoushik
"""
# Generic python libraries
import time
import os
import logging
import warnings
import random
import pickle
import datetime as dt
import pandas as pd
import numpy as np

# Creating GUI with tkinter
from tkinter import *
import tkinter

# Create Flask environment to expose bot as aservice
import json
from flask import Flask, render_template, request 
import requests

# Deep learning libraries
from keras.models import load_model
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


# To log the end to end process events

def log_process_activities():
    try:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.propagate = False
        log_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')

        log_handler = logging.FileHandler(
            'IAM_Chatbot_Logs.txt', mode="a", encoding=None, delay=False)
        log_handler.setFormatter(log_format)
        logger.addHandler(log_handler)
        return logger
    except Exception as e:
        #        print("Error at log_process ",e)
        logger.error("Error at log_process_activities ", e)


logger = log_process_activities()
