from wtforms import Form, TextAreaField, SubmitField, FileField, validators, RadioField
from spamfilter import app
import os
#https://wtforms.readthedocs.io/en/2.3.x/fields/
class InputForm(Form):
    '''
    Include 4 fields : 1. inputemail - a Text Area Field
                       2. inputfile - a File Field
                       3. inputmodel - a Radio Button
                       4. submit - a Submit Button
    '''
    files = [filename.rsplit('.', 1)[0] for filename in os.listdir(app.config["ML_MODEL_UPLOAD_FOLDER"])]
    inputemail = TextAreaField("Enter emails to predict")#,[validators.DataRequired(message="[This field is required.]"), validators.Length(min=3, max=100, message="[Field must be between 3 and 100 characters long.]")])
    inputfile = FileField("Upload only txt file here")
    inputmodel = RadioField("Model File", choices=files)
    submit = SubmitField('Submit Email')
    
