from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)


def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, dish TEXT, calories REAL, portion TEXT, date TEXT)')
    conn.commit()
    conn.close()


@app.route('/log', methods=['POST'])
def add_log():
    data = request.json
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('INSERT INTO logs (dish, calories, portion, date) VALUES (?, ?, ?, ?)',
              (data['dish'], data['calories'], data['portion'], data['date']))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})


@app.route('/logs', methods=['GET'])
def get_logs():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT * FROM logs')
    rows = c.fetchall()
    conn.close()
    logs = [{'id': r[0], 'dish': r[1], 'calories': r[2],
             'portion': r[3], 'date': r[4]} for r in rows]
    return jsonify(logs)


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
