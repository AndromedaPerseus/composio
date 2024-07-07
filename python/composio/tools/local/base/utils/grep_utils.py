#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path

import pathspec

from .grep_ast import TreeContext
from .parser import PARSERS, filename_to_lang


def get_files_excluding_gitignore(root_path, no_gitignore=False):
    """
    Get all files in the given root path, excluding those specified in .gitignore.

    :param root_path: The root directory to start searching from.
    :param no_gitignore: If True, ignore .gitignore file.
    :return: A list of file paths.
    """
    root_path = Path(root_path).resolve()
    gitignore = None

    if not no_gitignore:
        for parent in root_path.parents:
            potential_gitignore = parent / ".gitignore"
            if potential_gitignore.exists():
                gitignore = potential_gitignore
                break

    if gitignore:
        with open(gitignore, "r") as f:
            spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
    else:
        spec = pathspec.PathSpec.from_lines("gitwildmatch", [])

    files = []
    for path in root_path.rglob("*"):
        # Exclude .git and other version control system folders
        if any(part.startswith(".") and part != "." for part in path.parts):
            continue
        if path.is_file() and not spec.match_file(path):
            files.append(str(path))

    return files


# callable utility which works the same way as main.
def grep_util(
    pattern,
    filenames,
    encoding="utf8",
    color=None,
    verbose=False,
    line_number=True,
    ignore_case=True,
    no_gitignore=False,
):
    results = []

    for filename in filenames:
        if os.path.isdir(filename):
            dir_files = get_files_excluding_gitignore(filename, no_gitignore)
            for file in dir_files:
                results.extend(
                    process_file(
                        file,
                        pattern,
                        encoding,
                        ignore_case,
                        color,
                        verbose,
                        line_number,
                    )
                )
        else:
            results.extend(
                process_file(
                    filename,
                    pattern,
                    encoding,
                    ignore_case,
                    color,
                    verbose,
                    line_number,
                )
            )

    return results


def process_file(filename, pattern, encoding, ignore_case, color, verbose, line_number):
    file_results = []
    try:
        with open(filename, "r", encoding=encoding) as f:
            content = f.read()
    except UnicodeDecodeError:
        return file_results

    lang = filename_to_lang(filename)

    if lang:
        try:
            tc = TreeContext(
                filename, content, color=color, verbose=verbose, line_number=line_number
            )
            loi = tc.grep(pattern, ignore_case)
            if loi:
                tc.add_lines_of_interest(loi)
                tc.add_context()
                file_results.append({"filename": filename, "matches": tc.format()})
        except ValueError:
            pass  # Skip files that can't be parsed

    return file_results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", nargs="?", help="the pattern to search for")
    parser.add_argument(
        "filenames", nargs="*", help="the files to display", default="."
    )
    parser.add_argument("--encoding", default="utf8", help="file encoding")
    parser.add_argument(
        "--languages", action="store_true", help="show supported languages"
    )
    parser.add_argument(
        "-i", "--ignore-case", action="store_true", help="ignore case distinctions"
    )
    parser.add_argument(
        "--color", action="store_true", help="force color printing", default=None
    )
    parser.add_argument(
        "--no-color", action="store_false", help="disable color printing", dest="color"
    )
    parser.add_argument(
        "--no-gitignore", action="store_true", help="ignore .gitignore file"
    )
    parser.add_argument("--verbose", action="store_true", help="enable verbose output")
    parser.add_argument(
        "-n", "--line-number", action="store_true", help="display line numbers"
    )
    args = parser.parse_args()

    # If stdout is not a terminal, set color to False
    if args.color is None:
        args.color = os.isatty(1)

    # If --languages is provided, print the parsers table and exit
    if args.languages:
        for ext, lang in sorted(PARSERS.items()):
            print(f"{ext}: {lang}")
        return
    elif not args.pattern:
        print("Please provide a pattern to search for")
        return 1

    files_to_search = []
    for fname in args.filenames:
        if os.path.isdir(fname):
            files_to_search.extend(
                get_files_excluding_gitignore(fname, args.no_gitignore)
            )
        else:
            files_to_search.append(fname)

    for fname in files_to_search:
        process_filename(fname, args)


def process_filename(filename, args):
    try:
        with open(filename, "r", encoding=args.encoding) as file:
            code = file.read()
    except UnicodeDecodeError:
        return

    try:
        lang = filename_to_lang(filename)
        if lang is None:
            return
        tc = TreeContext(
            filename,
            code,
            color=args.color,
            verbose=args.verbose,
            line_number=args.line_number,
        )
    except ValueError:
        return

    loi = tc.grep(args.pattern, args.ignore_case)
    if not loi:
        return

    tc.add_lines_of_interest(loi)
    tc.add_context()

    print()
    print(f"{filename}:")

    print(tc.format(), end="")

    print()


if __name__ == "__main__":
    res = main()
    sys.exit(res)
