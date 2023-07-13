import os
import os.path as osp
import pickle
import argparse
import numpy as np


def check_info(info_dir):
    """Check the data in the info pickle file.
    """
    assert osp.exists(info_dir), f"Info file {info_dir} does not exists!"
    with open(info_dir, "rb") as handle:
        info = pickle.load(handle)
    
    print_str = ""
    for key in info.keys():
        val = info[key]
        if isinstance(val, (np.ndarray, np.generic)):
            if len(val.shape) > 1:
                print_str += f"{key}: array with shape {val.shape}\n"
            else:
                print_str += f"{key}: {val}\n"
        elif isinstance(val, list):
            if len(val) < 3:
                 print_str += f"{key}: {val}\n"
            elif isinstance(val[0], int):
                print_str += f"{key}: list with shape {len(val)}: {val}\n"
            else:
                print_str += f"{key}: list with shape {len(val)}\n"
        else:
            print_str += f"{key}: {val}\n"
    
    print(print_str)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info_dir", default="data/data_info/endonerf/pulling_soft_tissues.pkl",
                        type=str, help="Directory to info file.")
    args = parser.parse_args()
    
    check_info(args.info_dir)
    
    
if __name__ == "__main__":
    main()