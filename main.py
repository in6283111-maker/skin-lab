"""Точка входа для PyCharm: совпадает с командой `python skin_tagger.py`."""

from skin_tagger import parser, predict, train


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.command == "train":
        train(arguments)
    else:
        predict(arguments)
