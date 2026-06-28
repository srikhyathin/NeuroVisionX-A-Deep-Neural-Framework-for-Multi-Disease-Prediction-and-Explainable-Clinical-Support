import sqlite3
from datetime import datetime


class NeuroVisionDatabase:

    def __init__(self, db_name="neurovisionx.db"):

        self.connection = sqlite3.connect(
            db_name,
            check_same_thread=False
        )

        self.cursor = self.connection.cursor()

        self.create_tables()

    # -------------------------------------------------------

    def create_tables(self):

        self.cursor.execute("""

        CREATE TABLE IF NOT EXISTS patient_history(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            patient_name TEXT,

            age INTEGER,

            gender TEXT,

            disease TEXT,

            confidence REAL,

            model_used TEXT,

            symptoms TEXT,

            report_path TEXT,

            prediction_time TEXT

        )

        """)

        self.connection.commit()

    # -------------------------------------------------------

    def save_prediction(

        self,

        patient_name,

        age,

        gender,

        disease,

        confidence,

        model_used,

        symptoms,

        report_path

    ):

        self.cursor.execute("""

        INSERT INTO patient_history(

            patient_name,

            age,

            gender,

            disease,

            confidence,

            model_used,

            symptoms,

            report_path,

            prediction_time

        )

        VALUES(?,?,?,?,?,?,?,?,?)

        """,

        (

            patient_name,

            age,

            gender,

            disease,

            confidence,

            model_used,

            ",".join(symptoms),

            report_path,

            datetime.now().strftime(

                "%d-%m-%Y %H:%M:%S"

            )

        )

        )

        self.connection.commit()

    # -------------------------------------------------------

    def get_all_predictions(self):

        self.cursor.execute("""

        SELECT * FROM patient_history

        ORDER BY id DESC

        """)

        return self.cursor.fetchall()

    # -------------------------------------------------------

    def search_patient(self, patient_name):

        self.cursor.execute("""

        SELECT *

        FROM patient_history

        WHERE patient_name LIKE ?

        """,

        (

            "%" + patient_name + "%",

        )

        )

        return self.cursor.fetchall()

    # -------------------------------------------------------

    def delete_prediction(self, record_id):

        self.cursor.execute("""

        DELETE FROM patient_history

        WHERE id=?

        """,

        (

            record_id,

        )

        )

        self.connection.commit()

    # -------------------------------------------------------

    def total_predictions(self):

        self.cursor.execute("""

        SELECT COUNT(*)

        FROM patient_history

        """)

        return self.cursor.fetchone()[0]

    # -------------------------------------------------------

    def close(self):

        self.connection.close()