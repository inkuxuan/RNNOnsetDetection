import os
import re
import sys

if __name__ == '__main__':
    input_file_path = sys.argv[1]
    with open(input_file_path, 'r') as report_file:
        filename = os.path.basename(input_file_path)
        with open(filename+'.csv', 'w') as csv_file:
            csv_file.write("Threshold,Precision,Recall,F1\n")
            line = report_file.readline()
            while line:
                threshold_search = re.search("Height\\(Lambda\\)=([01]\\.\\d+)", line)
                if threshold_search:
                    threshold_now = threshold_search.group(1)
                    line = report_file.readline()
                    assert line
                    stats_search = re.search("Precision:([01]\\.\\d+) Recall:([01]\\.\\d+) F-score:([01]\\.\\d+)", line)
                    precision = stats_search.group(1)
                    recall = stats_search.group(2)
                    f1 = stats_search.group(3)
                    csv_file.write(f"{threshold_now},{precision},{recall},{f1}\n")
                line = report_file.readline()
