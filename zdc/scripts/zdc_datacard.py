#!/usr/bin/env python
import argparse
import zdc.modules.datacard as dc_tools

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inpath", type=str, help="Input path")
    parser.add_argument("cfgpath", type=str, help="Input config yaml")
    parser.add_argument("-d", "--dc", type=str, default="datacard.txt",
                        help="Output path for the datacard")
    parser.add_argument("-s", "--shape", type=str, default="shape.h5",
                        help="Output path for the shape file")
    parser.add_argument("--draw", default=False, action='store_true',
                        help="Draw alternative templates")
    return parser.parse_args()

def main():
    options = parse_args()
    dc_tools.datacard(**options)

if __name__ == "__main__":
    main()
