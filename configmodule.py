import os
# This is just a dictionary to save sensetive information like db logins etc
# access it using app.config[variable_name]

basedir = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER ="spamfilter/inputdata/"
ALLOWED_EXTENSIONS=["csv","xlsx","pdf"] # for further applications, just as per requirement
SECRET_KEY='prepord'
DEBUG=True
ML_MODEL_UPLOAD_FOLDER="spamfilter/mlmodels/"

"""
class Config(object):
    DEBUG = False
    ALLOWED_EXTENSIONS=["csv","xlsx","pdf"]
    TESTING = False
    ML_MODEL_UPLOAD_FOLDER="spamfilter\\mlmodels\\"
    UPLOAD_FOLDER ="spamfilter\\inputdata\\"
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or\
                             "sqlite:///" + os.path.join(basedir,"hello.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.environ.get("SECRET_KEY") or "your secret key"
    UPLOAD_FOLDER = "spamflter\\inputdata\\"
    ALLOWED_EXTENSIONS = ["csv","xlsx","pdf"] # for further applications, just as per requirement
    DB_NAME = "root"
    SESSION_COOKIE_SECURE = False

class TestingConfig(Config): #this overrides the default variables in Config class
    DEBUG=1
    TESTING=True
    FLASK_ENV="testing"
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(basedir,"testhello.db")
    #UPLOAD_FOLDER = "spamflter//inputdata" # in testing directory  --- to recive request.files and save here
    #ML_MODEL_UPLOAD_FOLDER="spamfilter\\mlmodels\\"
    SESSION_COOKIE_SECURE = False
    

class ProductionConfig(Config):
    SESSION_COOKIE_SECURE = True
    SECRET_KEY = "Production"
"""
