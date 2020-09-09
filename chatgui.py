"""
Created on Tue Aug 18 17:12:05 2020

@author: SaiKoushik
"""

from imports import *
from train_chatbot import train_bot
# To ignore deprication warnings of packages
warnings.filterwarnings("ignore", category=Warning)


def clean_up_sentence(sentence):
    """
    This function will clean up the input text and returns list of words

    Parameters
    ----------
    sentence : STRING
        DESCRIPTION: Variable contains user asked question.

    Returns
    -------
    sentence_words : LIST
        DESCRIPTION: Variable contains list of words from user question.

    """
    try:
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(
            word.lower()) for word in sentence_words]
        return sentence_words
    except Exception as e:
        logger.error("Error at clean_up_sentence ", e)


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    """
    This function prepocess the question asked by user and tokenize the pattern. 

    Parameters
    ----------
    sentence : STRING
        DESCRIPTION: Variable contains user asked question.
    words : pickle file.
        DESCRIPTION: Bot conversation words binary file.
    show_details : Flag, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    try:
        sentence_words = clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        pass
#                        print ("found in bag: %s" % w)
        return np.array(bag)
    except Exception as e:
        logger.error("Error at bow ", e)


def predict_class(sentence, model, words, classes):
    """
    This function predicts the class that bot will answers on the questions imposed by user.

    Parameters
    ----------
    sentence : STRING
        DESCRIPTION: Variable contains user asked question.
    model : h5 file
        DESCRIPTION: Bot trained binary file.
    words : pickle file.
        DESCRIPTION: Bot conversation words binary file.
    classes : pickle file.
        DESCRIPTION: Bot conversation classes binary file.

    Returns
    -------
    return_list : TYPE
        DESCRIPTION.

    """
    # filter out predictions below a threshold
    try:
        p = bow(sentence, words, show_details=False)
        # import pdb
        # pdb.set_trace()
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append(
                {"purpose": classes[r[0]], "probability": str(r[1])})
        return return_list
    except Exception as e:
        logger.error("Error at predict_class ", e)


def getResponse(ints, purposes_json):
    try:
        # import pdb
        # pdb.set_trace()
        tag = ints[0]['purpose']
        list_of_purposes = purposes_json['purposes']
        for i in list_of_purposes:
            if i['tag'] == tag:
                res_tag = i['tag']
                result = random.choice(i['responses'])
                res_context = i['context']
                break
        return res_tag, result, res_context
    except Exception as e:
        logger.error("Error at getResponse ", e)


def chatbot_response(msg, model, purposes, words, classes):
    """
    This function will be called when bot communicates using flask api.

    Parameters
    ----------
    msg : STRING
        DESCRIPTION: Variable contains user asked question.
    model : h5 file
        DESCRIPTION: Bot trained binary file.
    purposes : JSON
        DESCRIPTION: JOSN contains domine specific Questioniers.
    words : pickle file.
        DESCRIPTION: Bot conversation words binary file.
    classes : pickle file.
        DESCRIPTION: Bot conversation classes binary file.

    Returns
    -------
    res : STRING
        DESCRIPTION: Actual bot predicted answer to the conversation.

    """
    try:
        # import pdb
        # pdb.set_trace()
        ints = predict_class(msg, model, words, classes)
        res_tag, res, res_context = getResponse(ints, purposes)
        if 'ambiguous_response' in res_context:
            save_ambiguous_chat(msg, res, res_tag, 'From_API')
        # return res_tag, res, res_context
        # print("******** I'm here hellooo")
        return str(res)
    except Exception as e:
        logger.error("Error at chatbot_response ", e)


def chatbot_response_tk_window(msg):
    """
    This function will be called when bot communicate using TKInter GUI.

    Parameters
    ----------
    msg : TYPE
        DESCRIPTION: Variable contains user asked question.

    Returns
    -------
    res_tag : STRING
        DESCRIPTION: Actual bot predicted tag to the conversation.
    res : STRING
        DESCRIPTION: Actual bot predicted answer to the conversation.
    res_context : STRING
        DESCRIPTION: Actual bot predicted Context to the conversation.

    """
    try:
        # import pdb
        # pdb.set_trace()
        ints = predict_class(msg, model, words, classes)
        res_tag, res, res_context = getResponse(ints, purposes)
        if 'ambiguous_response' in res_context:
            save_ambiguous_chat(msg, res, res_tag, 'From_TK')
        # return res_tag, res, res_context
        # print("******** I'm here hellooo")
        return res_tag, res, res_context
    except Exception as e:
        logger.error("Error at chatbot_response ", e)


def save_ambiguous_chat(msg, res, res_tag, Chat_From):
    """
    This function will save the ambigueous conversation between bot and user.

    Parameters
    ----------
    msg : STRING
        DESCRIPTION: Variable contains user asked question.
    res : STRING
        DESCRIPTION: Variable contains bot answer.
    res_tag : STRING
        DESCRIPTION: Variable contains tag that the response belongs to.
    Chat_From : STRING
        DESCRIPTION: Variable contains communication bot communication channel.

    Returns
    -------
    None.

    """
    try:
        amb_dict = {"Responded Time": [dt.datetime.now()], "Chat_From": [Chat_From],
                    "Control/Tag": [res_tag], "User Query": [msg],
                    "Bot Response": [res], "Intended Answer": [""]}

        df = pd.DataFrame(amb_dict)

        f_name = "ambiguous_bot_conversation.csv"
        if os.path.exists(f_name):
            earlier_amb = pd.read_csv(f_name)
            earlier_amb = earlier_amb.append(df)
            earlier_amb.to_csv(f_name, index=False, header=True)
#            print("I'm here")
            logger.info(
                "ChatBot encountered ambiguity, hence logged such conversations")
        else:
            df.to_csv(f_name, index=False, header=True)
    except Exception as e:
        logger.error("Error at save_ambiguous_chat ", e)


def chatbot_convo():
    """
    This function will be called when bot tries to esablish communicate using TKInter window.

    Returns
    -------
    None.

    """
    try:
        def send():
            try:
                msg = EntryBox.get("1.0", 'end-1c').strip()
                EntryBox.delete("0.0", END)

                if msg != '':
                    ChatLog.config(state=NORMAL)
                    ChatLog.insert(END, "You: " + msg + '\n\n')
                    ChatLog.config(foreground="#442265", font=("Verdana", 12))

                    res_tag, res, res_context = chatbot_response_tk_window(
                        msg)  # ,model,purposes,words,classes)
                    # if 'ambiguous_response' in res_context:
                    #     save_ambiguous_chat(msg, res, res_tag)
                    ChatLog.insert(END, "Bot: " + res + '\n\n')

                    ChatLog.config(state=DISABLED)
                    ChatLog.yview(END)

            except Exception as e:
                logger.error("Error at send ", e)

        """create chat gui and make conversionsation"""
        base = Tk()
        base.title("IAM-Giene")
        base.geometry("400x500")
        base.resizable(width=FALSE, height=FALSE)

    # Create Chat window
        ChatLog = Text(base, bd=0, bg="white", height="8",
                       width="50", font="Arial",)
        ChatLog.config(state=DISABLED)

    # Bind scrollbar to Chat window
        scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
        ChatLog['yscrollcommand'] = scrollbar.set

        SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                            bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                            command=send)

    # Create the box to enter message
        EntryBox = Text(base, bd=0, bg="white", width="29",
                        height="5", font="Arial")

    # Place all components on the screen
        scrollbar.place(x=376, y=6, height=386)
        ChatLog.place(x=6, y=6, height=386, width=370)
        EntryBox.place(x=128, y=401, height=90, width=265)
        SendButton.place(x=6, y=401, height=90)

        base.mainloop()
    except Exception as e:
        logger.error("Error at chatbot_convo ", e)


if __name__ == "__main__":
    # """ train or retrain the model"""

    train_bot()
    logger.info("Giene Communication Started")
    model = load_model('chatbot_model.h5')
    purposes = json.loads(open('purposes.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    chatbot_convo()

    logger.info(
        'IAM-Giene communicated  on : %s ', dt.datetime.now())
