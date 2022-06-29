import sys
import re


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} filename")
        exit(1)
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if "Total" in line]
    if len(lines) == 0:
        print("Bad filename; check your args")
        exit(1)
    pat = re.compile(r'\d[.]\d+')
    seconds = [float(re.findall(pat, line)[0]) for line in lines]
    diffs = []
    num_outliers = 0
    for i in range(0, len(seconds), 2):
        diff = seconds[i] - seconds[i+1]
        if diff >= 0.5:
            diffs.append(diff)
        else:
            num_outliers += 1
    print(f"Erased {num_outliers} outliers")
    separator = '\r\n'
    with open(f'filtered_{filename}', 'w') as f:
        for result in diffs:
            f.write(str(result))
            f.write(separator)


if __name__ == "__main__":
    main()

