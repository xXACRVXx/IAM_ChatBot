"""
Created on Tue Aug 18 17:12:05 2020

@author: SaiKoushik
"""

from imports import *
# To ignore deprication warnings of packages
warnings.filterwarnings("ignore", category=Warning)


def on_purposes_file_update():
    """
    This function try to update the purposes.json file on timely basis.

    Returns
    -------
    str
        DESCRIPTION.

    """
    try:
        last_modified_date = time.ctime(os.path.getmtime("purposes.json"))
        last_modified_date = dt.datetime.strptime(
            last_modified_date, '%a %b %d %H:%M:%S %Y').strftime('%Y-%m-%d')
        delta = (dt.datetime.strptime(str(last_modified_date), "%Y-%m-%d") -
                 dt.datetime.strptime(str(dt.datetime.today().strftime('%Y-%m-%d')), '%Y-%m-%d'))
        if delta.days < 7:
            return 'YES'
        else:
            return "NO"
    except Exception as e:
        logger.error("Error at on_purposes_file_update ", e)


def on_chatbot_model_file_update():
    """
    This function try to retrain the chatbot_model.h5 file on timely basis.

    Returns
    -------
    str
        DESCRIPTION.

    """
    try:
        last_modified_date = time.ctime(os.path.getmtime("chatbot_model.h5"))
        last_modified_date = dt.datetime.strptime(
            last_modified_date, '%a %b %d %H:%M:%S %Y').strftime('%Y-%m-%d')
        delta = (dt.datetime.strptime(str(last_modified_date), "%Y-%m-%d") -
                 dt.datetime.strptime(str(dt.datetime.today().strftime('%Y-%m-%d')), '%Y-%m-%d'))
        if delta.days > 15:
            return 'YES'
        else:
            return "NO"
    except Exception as e:
        logger.error("Error at on_chatbot_model_file_update ", e)


def retrain_chatbot():
    """
    This function helps in retraining the chatbot_model.h5 file and make bot communicate effectivly.

    Returns
    -------
    None.

    """
    try:
        words = []
        classes = []
        documents = []
        ignore_words = ['?', '!']
        data_file = open('purposes.json').read()
        purposes = json.loads(data_file)

        for purpose in purposes['purposes']:
            for pattern in purpose['patterns']:

                # take each word and tokenize it
                w = nltk.word_tokenize(pattern)
                words.extend(w)
                # adding documents
                documents.append((w, purpose['tag']))

                # adding classes to our class list
                if purpose['tag'] not in classes:
                    classes.append(purpose['tag'])

        words = [lemmatizer.lemmatize(w.lower())
                 for w in words if w not in ignore_words]
        words = sorted(list(set(words)))

        classes = sorted(list(set(classes)))

#        print(len(documents), "documents")
#        print(len(classes), "classes", classes)
#        print(len(words), "unique lemmatized words", words)

#        pickle.dump(words,open('words.pkl','wb'))
#        pickle.dump(classes,open('classes.pkl','wb'))

        # initializing training data
        training = []
        output_empty = [0] * len(classes)
        for doc in documents:
            # initializing bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # lemmatize each word - create base word, in attempt to represent related words
            pattern_words = [lemmatizer.lemmatize(
                word.lower()) for word in pattern_words]
            # create our bag of words array with 1, if word match found in current pattern
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            # output is a '0' for each tag and '1' for current tag (for each pattern)
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1

            training.append([bag, output_row])
        # shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training)
        # create train and test lists. X - patterns, Y - purposes
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
#        print("Training data created!")

        # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
        # equal to number of purposes to predict output purpose with softmax
        model = Sequential()
        model.add(Dense(128, input_shape=(
            len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))

        # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd, metrics=['accuracy'])

        # fitting and saving the model
        hist = model.fit(np.array(train_x), np.array(train_y),
                         epochs=200, batch_size=5, verbose=1)
        model.save('chatbot_model.h5', hist)
        pickle.dump(words, open('words.pkl', 'wb'))
        pickle.dump(classes, open('classes.pkl', 'wb'))
        logger.info("chatbot model created!")
    except Exception as e:
        logger.error("Error at retrain_chatbot ", e)


def train_bot():
    """
    This function enable to retrain/train BOT on timely basis.

    Returns
    -------
    None.

    """
    try:
        #        logger=log_process_activities()
        check_on_purposes_file_update = on_purposes_file_update()
        checkon_chatbot_model_file_update = on_chatbot_model_file_update()
        if check_on_purposes_file_update == "YES" or checkon_chatbot_model_file_update == "YES":
            #            print("retrain model")
            retrain_chatbot()
    except Exception as e:
        logger.error("Error at train_bot ", e)
