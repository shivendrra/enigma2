import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

from enigma2 import Database

db = Database(topics=["Human"], out_dir="./data/", mode='csv', email="shivharsh44@gmail.com", retmax=100)
db.build()