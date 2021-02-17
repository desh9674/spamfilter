the application folder is named spamfilter
before that folder there is a run.py file which sets the app as spamfilter folder

runs commands in cmd to run the application:

first change directory
cd C:\Users\1735742\Documents\Env\caps1

set FLASK_APP=run.py  
set FLASK_DBUG=1   / debugging true
set FLASK_ENV= testing / 

flask run   / run application

#your routes and flask blueprint are defined in spamfilter_api.py
# the __init__ has app configurations and package
#