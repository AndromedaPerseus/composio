# Import necessary libraries
import colorsys
import math
import os
import random
import sys
import warnings
from collections import Counter, defaultdict, namedtuple
from importlib import resources
from pathlib import Path
import shutil
from diskcache import Cache
from composio.tools.local.base.utils.parser import filename_to_lang, PARSERS
from composio.tools.local.base.utils.grep_ast import TreeContext
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tqdm import tqdm

# Suppress FutureWarning from tree_sitter
warnings.simplefilter("ignore", category=FutureWarning)
from tree_sitter_languages import get_language, get_parser  # noqa: E402

# Define a named tuple for storing tag information
Tag = namedtuple("Tag", "rel_fname fname line name kind".split())

"""
RepoMap: Generates a structured view of a code repository.

Key components:
1. chat_fnames: Files active in the current conversation.
   - Given high personalization priority in ranking.
   - Excluded from final repo map output.

2. mentioned_fnames: Files of particular interest.
   - Given medium personalization priority in ranking.
   - Included in final repo map output.

3. other_fnames: All other repository files.
   - Given low personalization priority.
   - Included in output if relevant or space allows.

Process:
1. Collect tags (definitions and references) from all files.
2. Build a graph representing relationships between files and identifiers.
3. Use PageRank with personalization to rank files and identifiers.
4. Generate a tree-like structure of the most important elements.
5. Optimize the output to fit within a specified token limit.

The resulting repo map provides a context-aware overview of the repository,
emphasizing relevant files and code structures based on the current conversation
and mentioned points of interest.
"""

from transformers import GPT2Tokenizer


# Replace the existing token_count function with this:
def token_count(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    max_length = 1024
    # If the tokens exceed max_length, count them in chunks
    if len(text) > max_length:
        total_tokens = 0
        for i in range(0, len(text), max_length):
            chunk = tokenizer.encode(text[i : i + max_length])
            total_tokens += len(chunk)
        return total_tokens
    else:
        return len(tokenizer.encode(text))


def print_if_verbose(text, verbose=True):
    if verbose:
        print(text)


class RepoMap:
    # Class variables for caching
    CACHE_VERSION = 1
    TAGS_CACHE_DIR = f".composio.tags.cache.v{CACHE_VERSION}"

    cache_missing = False
    warned_files = set()

    def __init__(
        self,
        map_tokens=10240,
        root=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=10000,
    ):
        """
        Initialize the RepoMap object.

        :param map_tokens: Maximum number of tokens for the repo map
        :param root: Root directory of the repository
        :param repo_content_prefix: Prefix for repo content
        :param verbose: Enable verbose output
        :param max_context_window: Maximum context window size
        """

        self.verbose = verbose

        # Set root directory
        if not root:
            root = os.getcwd()
        self.root = root

        # Load tags cache
        self.load_tags_cache()

        self.max_map_tokens = map_tokens
        self.max_context_window = max_context_window

        self.token_count = token_count
        self.repo_content_prefix = repo_content_prefix

    def get_repo_map(
        self, chat_files, other_files, mentioned_fnames=None, mentioned_idents=None
    ):
        """
        Generate a repository map based on the given files and mentions.

        :param chat_files: Files currently in the chat
        :param other_files: Other files in the repository
        :param mentioned_fnames: Mentioned file names
        :param mentioned_idents: Mentioned identifiers
        :return: Repository map as a string
        """
        # Early exit conditions
        if self.max_map_tokens <= 0 or not other_files:
            print_if_verbose(
                "Exiting repo-map due to max_map_tokens <= 0 or no other_files",
                self.verbose,
            )
            return

        # Initialize mentioned sets if not provided
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        max_map_tokens = self.max_map_tokens

        # Adjust max_map_tokens when no files are in the chat
        MUL = 8
        padding = 4096
        if max_map_tokens and self.max_context_window:
            target = min(max_map_tokens * MUL, self.max_context_window - padding)
        else:
            target = 0
        if not chat_files and self.max_context_window and target > 0:
            max_map_tokens = target

        try:
            # Generate ranked tags map
            files_listing = self.get_ranked_tags_map(
                chat_files,
                other_files,
                max_map_tokens,
                mentioned_fnames,
                mentioned_idents,
            )
        except RecursionError:
            # Handle recursion error (possibly due to large git repo)
            self.max_map_tokens = 0
            print_if_verbose("Exiting repo-map due to RecursionError", self.verbose)
            return

        if not files_listing:
            print_if_verbose(
                "Exiting repo-map due to empty files_listing", self.verbose
            )
            return

        # Count tokens in the files listing
        num_tokens = self.token_count(files_listing)
        print_if_verbose(f"Repo-map: {num_tokens/1024:.1f} k-tokens", self.verbose)

        # Prepare repo content string
        other = "other " if chat_files else ""
        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""

        repo_content += files_listing

        return repo_content

    def get_rel_fname(self, fname):
        """Get relative file name from the root directory."""
        return os.path.relpath(fname, self.root)

    def split_path(self, path):
        """Split path into components relative to the root directory."""
        path = os.path.relpath(path, self.root)
        return [path + ":"]

    def load_tags_cache(self):
        """Load tags cache from disk."""
        path = Path(self.root) / self.TAGS_CACHE_DIR
        if not path.exists():
            self.cache_missing = True
        self.TAGS_CACHE = Cache(path)

    def save_tags_cache(self):
        """Save tags cache to disk (placeholder method)."""
        pass

    def get_mtime(self, fname):
        """Get modification time of a file."""
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            print_if_verbose(f"File not found error: {fname}", self.verbose)

    def get_tags(self, fname: str, rel_fname: str) -> list[Tag]:
        """
        Get tags for a file, using cache if available.

        :param fname: Absolute file name
        :param rel_fname: Relative file name
        :return: List of tags
        """
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            print_if_verbose(
                f"Warning: Unable to get modification time for {fname}, skipping",
                self.verbose,
            )
            return []

        if self.TAGS_CACHE is None:
            print_if_verbose(
                "Warning: Tags cache is not initialized, something went wrong",
                self.verbose,
            )
            return []

        cache_key = fname
        cache_data = self.TAGS_CACHE.get(cache_key)

        if cache_data is not None and cache_data.get("mtime") == file_mtime:
            return cache_data.get("data", [])

        # Cache miss or outdated: generate new tags
        data = list(self.get_tags_raw(fname, rel_fname))

        # Update cache
        self.TAGS_CACHE.set(cache_key, {"mtime": file_mtime, "data": data})
        self.save_tags_cache()

        return data

    def get_tags_raw(self, fname, rel_fname):
        """
        Generate tags for a file using tree-sitter and pygments.

        :param fname: Absolute file name
        :param rel_fname: Relative file name
        :yield: Tag objects
        """
        # Determine language and get parser
        lang = filename_to_lang(fname)
        if not lang:
            print_if_verbose(
                "Exiting get_tags_raw due to no language detected", self.verbose
            )
            return

        language = get_language(lang)
        parser = get_parser(lang)

        # Load tags query
        try:
            scm_fname = resources.files(__package__).joinpath(
                "queries", f"tree-sitter-{lang}-tags.scm"
            )
        except KeyError:
            print_if_verbose(
                "Exiting get_tags_raw due to KeyError in loading tags query",
                self.verbose,
            )
            return

        if not scm_fname.is_file():
            print_if_verbose(
                "Exiting get_tags_raw due to non-existent query_scm", self.verbose
            )
            return
        query_scm = scm_fname.read_text()

        # Parse code
        code = None

        with open(fname, "r", encoding="utf-8") as file:
            code = file.read()
            code = code.strip()

        if not code or len(code) == 0:
            print_if_verbose("Exiting get_tags_raw due to empty code", self.verbose)
            return

        tree = parser.parse(bytes(code, "utf-8"))

        # Run tags query
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)

        captures = list(captures)

        saw = set()
        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind = "def"
            elif tag.startswith("name.reference."):
                kind = "ref"
            else:
                continue

            saw.add(kind)

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node.text.decode("utf-8"),
                kind=kind,
                line=node.start_point[0],
            )

            yield result
        # If no references found, use pygments for additional tagging
        if "ref" in saw:
            print_if_verbose(
                "Exiting get_tags_raw after processing references", self.verbose
            )
            # print file name
            print_if_verbose(fname, self.verbose)
            return
        if "def" not in saw:
            print_if_verbose(
                "Exiting get_tags_raw due to no definitions found", self.verbose
            )
            return

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except ClassNotFound:
            print_if_verbose(
                "Exiting get_tags_raw due to ClassNotFound in lexer guessing",
                self.verbose,
            )
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
            )

    def get_ranked_tags(
        self, chat_fnames, other_fnames, mentioned_fnames, mentioned_idents
    ):
        """
        Generate ranked tags for files in the repository.

        :param chat_fnames: Files currently in the chat
        :param other_fnames: Other files in the repository
        :param mentioned_fnames: Mentioned file names
        :param mentioned_idents: Mentioned identifiers
        :return: List of ranked tags
        """
        import networkx as nx

        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)

        personalization = dict()

        fnames = set(chat_fnames).union(set(other_fnames))
        chat_rel_fnames = set()

        fnames = sorted(fnames)

        # Improved personalization logic
        chat_weight = 10.0
        mentioned_weight = 15.0
        other_weight = 1.0

        total_weight = (
            len(chat_fnames) * chat_weight
            + len(mentioned_fnames) * mentioned_weight
            + len(other_fnames) * other_weight
        )

        if self.cache_missing:
            fnames = tqdm(fnames)
        self.cache_missing = False

        # Process each file
        for fname in fnames:
            if not Path(fname).is_file():
                if fname not in self.warned_files:
                    if Path(fname).exists():
                        print_if_verbose(
                            f"Repo-map can't include {fname}, it is not a normal file",
                            self.verbose,
                        )
                    else:
                        print_if_verbose(
                            f"Repo-map can't include {fname}, it no longer exists",
                            self.verbose,
                        )

                self.warned_files.add(fname)
                continue

            rel_fname = self.get_rel_fname(fname)

            if fname in chat_fnames:
                personalization[rel_fname] = chat_weight / total_weight
                chat_rel_fnames.add(rel_fname)
            elif rel_fname in mentioned_fnames:
                personalization[rel_fname] = mentioned_weight / total_weight
            else:
                personalization[rel_fname] = other_weight / total_weight

            tags = self.get_tags(fname, rel_fname)
            if tags is None:
                continue

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)

                if tag.kind == "ref":
                    references[tag.name].append(rel_fname)

        # If no references, use definitions as references
        if not references:
            references = dict((k, list(v)) for k, v in defines.items())

        idents = set(defines.keys()).intersection(set(references.keys()))

        # Create graph
        G = nx.MultiDiGraph()

        for ident in idents:
            definers = defines[ident]
            if ident in mentioned_idents:
                mul = 2.0
            elif ident.startswith("_"):
                mul = 0.5
            else:
                mul = 1.0

            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    # Scale down high-frequency mentions
                    num_refs = math.sqrt(num_refs)

                    G.add_edge(referencer, definer, weight=mul * num_refs, ident=ident)

        # Calculate PageRank
        try:
            ranked = nx.pagerank(G, weight="weight", personalization=personalization)
        except ZeroDivisionError:
            print_if_verbose(
                "Exiting get_ranked_tags due to ZeroDivisionError in PageRank calculation",
                self.verbose,
            )
            return []

        # Distribute rank across edges
        ranked_definitions = defaultdict(float)
        for src in G.nodes:
            src_rank = ranked[src]
            total_weight = sum(
                data["weight"] for _src, _dst, data in G.out_edges(src, data=True)
            )
            for _src, dst, data in G.out_edges(src, data=True):
                data["rank"] = src_rank * data["weight"] / total_weight
                ident = data["ident"]
                ranked_definitions[(dst, ident)] += data["rank"]

        ranked_tags = []
        ranked_definitions = sorted(
            ranked_definitions.items(), reverse=True, key=lambda x: x[1]
        )

        # Generate ranked tags
        for (fname, ident), rank in ranked_definitions:
            if fname in chat_rel_fnames:
                continue
            ranked_tags += list(definitions.get((fname, ident), []))

        rel_other_fnames_without_tags = set(
            self.get_rel_fname(fname) for fname in other_fnames
        )

        fnames_already_included = set(rt[0] for rt in ranked_tags)

        # Add remaining files to ranked tags
        top_rank = sorted(
            [(rank, node) for (node, rank) in ranked.items()], reverse=True
        )
        for rank, fname in top_rank:
            if fname in rel_other_fnames_without_tags:
                rel_other_fnames_without_tags.remove(fname)
            if fname not in fnames_already_included:
                ranked_tags.append((fname,))

        for fname in rel_other_fnames_without_tags:
            ranked_tags.append((fname,))

        return ranked_tags

    def get_ranked_tags_map(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
    ):
        """
        Generate a ranked tags map for the repository.

        :param chat_fnames: Files currently in the chat
        :param other_fnames: Other files in the repository
        :param max_map_tokens: Maximum number of tokens for the map
        :param mentioned_fnames: Mentioned file names
        :param mentioned_idents: Mentioned identifiers
        :return: Formatted string of the ranked tags map
        """
        if not other_fnames:
            other_fnames = list()
        if not max_map_tokens:
            max_map_tokens = self.max_map_tokens
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        ranked_tags = self.get_ranked_tags(
            chat_fnames, other_fnames, mentioned_fnames, mentioned_idents
        )

        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None
        best_tree_tokens = 0

        chat_rel_fnames = [self.get_rel_fname(fname) for fname in chat_fnames]

        # Binary search for optimal number of tags
        middle = min(max_map_tokens // 25, num_tags)

        self.tree_cache = dict()

        while lower_bound <= upper_bound:
            tree = self.to_tree(ranked_tags[:middle], chat_rel_fnames)
            num_tokens = self.token_count(tree)

            if num_tokens < max_map_tokens and num_tokens > best_tree_tokens:
                best_tree = tree
                best_tree_tokens = num_tokens

            if num_tokens < max_map_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = (lower_bound + upper_bound) // 2

        return best_tree

    tree_cache = dict()

    def render_tree(self, abs_fname, rel_fname, lois):
        key = (rel_fname, tuple(sorted(lois)))

        if key in self.tree_cache:
            return self.tree_cache[key]

        # use python to read the file
        with open(abs_fname, "r", encoding="utf-8") as file:
            code = file.read()
        if not code.endswith("\n"):
            code += "\n"

        context = TreeContext(
            rel_fname,
            code,
            color=False,
            line_number=False,
            child_context=False,
            last_line=False,
            margin=0,
            mark_lois=False,
            loi_pad=0,
            # header_max=30,
            show_top_of_file_parent_scope=False,
        )

        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res

    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        tags = [tag for tag in tags if tag[0] not in chat_rel_fnames]
        tags = sorted(tags)

        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        # add a bogus tag at the end so we trip the this_fname != cur_fname...
        dummy_tag = (None,)
        for tag in tags + [dummy_tag]:
            this_rel_fname = tag[0]

            # ... here ... to output the final real entry in the list
            if this_rel_fname != cur_fname:
                if lois is not None:
                    output += "\n"
                    if cur_fname is not None:
                        output += cur_fname + ":\n"
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                    lois = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"
                if type(tag) is Tag:
                    lois = []
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None:
                lois.append(tag.line)

        # truncate long lines, in case we get minified js or something else crazy
        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"

        return output

    def delete_cache(self):
        """Delete the tags cache."""
        cache_path = Path(self.root) / self.TAGS_CACHE_DIR
        print("Deleting cache: ", cache_path)
        if cache_path.exists():
            # Remove all files and subdirectories
            for item in cache_path.glob("*"):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            print(f"Cache contents deleted: {cache_path}")
        else:
            print("No cache found to delete.")

            # Reset the cache object
            self.TAGS_CACHE = Cache(cache_path)
            self.cache_missing = True


def find_src_files(directory):
    if not os.path.isdir(directory):
        return [directory]

    src_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            src_files.append(os.path.join(root, file))
    return src_files
