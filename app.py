from chatgui import chatbot_response
from train_chatbot import train_bot
from imports import *
# To ignore deprication warnings of packages
warnings.filterwarnings("ignore", category=Warning)

# Create Flask app instance#
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/IAMChatBot", methods=['GET', 'POST'])
def get_bot_response():

    # train_bot()
    logger.info("Giene Communication Started")
    model = load_model('chatbot_model.h5')
    purposes = json.loads(open('purposes.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    userText = request.args.get('msg')
    res = str(chatbot_response(userText, model, purposes, words, classes))
    logger.info("Giene Communication Ended")

    if userText == None:
        return render_template("index.html")
    else:
        return res


if __name__ == "__main__":

    app.run(debug=True)
    # app.run()
