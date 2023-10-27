import os
import sys
from ctk_cli import CLIArgumentParser

def main(args):  

    _ = os.system("printf '\n---\n\nFOUND: [{}]\n'".format(args.input_file))

    cwd = os.getcwd()
    print(cwd)
    os.chdir(cwd)
    
    cmd = "python3 ../dsacode/parse_args.py --input_file {} --base_dir {} --modelfile {} --girderApiUrl {} --girderToken {}".format(args.input_file, args.base_dir, args.modelfile, args.girderApiUrl, args.girderToken)
    print(cmd)
    sys.stdout.flush()
    os.system(cmd)  


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
    