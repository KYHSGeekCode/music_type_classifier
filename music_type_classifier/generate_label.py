# generate label (music_id -> type) from db.

import sqlite3


def get_labels() -> [int, int]:
    con = sqlite3.connect('../main.sqlite')
    cur = con.cursor()
    result = {}
    for row in cur.execute('''SELECT music_data_id, circle_type FROM live_data '''):
        result[row[0]] = row[1]
    cur.close()
    con.close()
    return result


if __name__ == '__main__':
    for r in get_labels():
        print(r)
