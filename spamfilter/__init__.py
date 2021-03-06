from flask import Flask
import os
#from spamfilter.spamfilter_api import spam_api

app = Flask(__name__)
app.config.from_object('configmodule')

# for class config use config.class_name-- eg config.TestingConfig

# Now we can access the configuration variables via app.config["VAR_NAME"].
#from app import views  # for user views.py
#from app import admin_views # for admin views with admin prefix url





def create_app(testing_config=None):

    app = Flask(__name__, instance_relative_config=True)

    #Setting default Settings to your application
    app.config.from_mapping(
        SECRET_KEY='dev',
        )

    if testing_config is None:
        # Overrides default settings based on settings defined in config.py file, present under application instance folder
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(testing_config)

    from spamfilter.models import db, migrate
    db.init_app(app)
    migrate.init_app(app, db)
    with app.app_context():
        from spamfilter.models import File
        db.create_all()

    app.register_blueprint(spam_api)
    @app.route('/home')
    def home():
        return 'This verifies Application working Status : '+app.config['SECRET_KEY']

    return app


