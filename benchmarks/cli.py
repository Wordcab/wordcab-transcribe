from argparse import ArgumentParser

from benchmarks.timestamps_comparison import TimestampsComparison


def main():
    parser = ArgumentParser("Wordcab-Transcribe CLI tools for benchmarking.", usage="wordcab-transcribe <command> [<args>]")
    commands_parser = parser.add_subparsers(help="wordcab-transcribe command helpers")

    # Register subcommands
    TimestampsComparison.register_subcommand(commands_parser)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    args.func(args).run()


if __name__ == "__main__":
    main()
