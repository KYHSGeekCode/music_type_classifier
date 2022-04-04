# generate label (music_id -> type) from db.

import sqlite3


def run():
    con = sqlite3.connect('../main.sqlite')
    cur = con.cursor()
    for row in cur.execute('''SELECT music_data_id, circle_type FROM live_data '''):
        print(row)
    cur.close()
    con.close()


if __name__ == '__main__':
    run()
