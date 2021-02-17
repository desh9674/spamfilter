from spamfilter import app
from spamfilter.spamfilter_api import spam_api

# Use your app molds here,  many apps and their respective APIs or Views
app.register_blueprint(spam_api, url_prefix="/")

