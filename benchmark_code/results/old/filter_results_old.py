import sys
import re


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} filename")
        exit(1)
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if "Total" not in line]
    pat = re.compile(r'\d[.]\d+')
    seconds = [re.findall(pat, line)[0] for line in lines]
    one_thread = seconds[:50]
    two_threads_tmp = seconds[50:]
    two_threads = []
    for i in range(0, len(two_threads_tmp), 2):
        two_threads.append(max(two_threads_tmp[i], two_threads_tmp[i+1]))
    separator = '\r\n'
    with open(f'filtered_{filename}', 'w') as f:
        for result in one_thread:
            f.write(result)
            f.write(separator)
        for result in two_threads:
            f.write(result)
            f.write(separator)


if __name__ == "__main__":
    main()

