import os
import argparse

def main(args):

    from reconstruct_code import predict

    print('inference starts')
    predict(args=args)
    print('inference ends')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--girderApiUrl', dest='girderApiUrl', default=' ' ,type=str,
        help='girderApiUrl')
    parser.add_argument('--girderToken', dest='girderToken', default=' ' ,type=str,
        help='girderToken')
    parser.add_argument('--base_dir', dest='base_dir', default=os.getcwd(),type=str,
        help='base directory of code folder')
    parser.add_argument('--modelfile', dest='modelfile', default=None ,type=str,
        help='the desired model file to use for training or prediction')
    parser.add_argument('--input_file', dest='input_file', default=' ' ,type=str,
        help='input_file')
    
    args = parser.parse_args()
    main(args=args)
