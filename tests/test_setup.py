import sys

import mysql.connector
import nltk
import pandas as pd


def test_python_version():
    assert sys.version_info >= (3, 11)


def test_packages_import():
    assert pd is not None
    assert nltk is not None


def test_mysql_connection():
    connection = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="root",
        database="real_estate",
    )

    cursor = connection.cursor()
    cursor.execute("SELECT 1")
    assert cursor.fetchone() == (1,)

    cursor.close()
    connection.close()
