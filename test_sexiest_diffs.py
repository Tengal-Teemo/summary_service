import os

from pprint import pprint

from sexiest_diffs import summary_from_files


def main():
    test_path = 'test_files'
    odt_v1_fp = 'v1.odt'
    odt_v2_fp = 'v2.odt'
    relevant_diffs = summary_from_files(os.path.join(test_path, odt_v1_fp), os.path.join(test_path, odt_v2_fp), 'Someone who loves the drums')
    pprint(relevant_diffs)


if __name__ == '__main__':
    main()
