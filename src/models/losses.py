import argparse
import logging

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main(args):
    logging.info(args.message)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--message", type=str, default="Hello World", help="model file")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
