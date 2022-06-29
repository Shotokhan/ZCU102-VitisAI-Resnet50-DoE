import csv
import re


def read_csv(filename):
    rows = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        [rows.append(row) for row in reader]
    return rows


def create_output(csv_rows, img_type):
    assert img_type in ["train", "test", "validation", "challenge2018"]
    # discard header
    csv_rows = csv_rows[1:]
    # the csv file has some repeated labels
    image_ids = set([row[0] for row in csv_rows])
    # it also has some invalid names
    pat = re.compile(r"^[a-f0-9]{1,}$")
    image_ids = [img_id for img_id in image_ids if re.match(pat, img_id)]
    # now I have to craft the well formatted output
    text_lines = [f"{img_type}/{img_id}" for img_id in image_ids]
    with open("images_to_download.txt", 'w') as f:
        for line in text_lines:
            f.write(line + '\n')
    return text_lines


if __name__ == "__main__":
    filename = "validation-annotations-human-imagelabels-boxable.csv"
    rows = read_csv(filename)
    create_output(rows, "validation")

