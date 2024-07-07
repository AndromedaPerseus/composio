import logging

from composio import Action, App, ComposioToolSet

logging.getLogger("chromadb").setLevel(logging.ERROR)

import multiprocessing
import os
import time


def my_func(input1: str, input2: str):
    print("My func")


class MyClass:
    def __init__(self, input1: str, input2: str):
        self.input1 = input1
        self.input2 = input2


def main():
    toolset = ComposioToolSet()

    # get current path
    current_path = os.getcwd()
    # get directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print("Dir Path: ", dir_path)
    # delete repo map cache
    # result = toolset.execute_action(
    #     action=Action.CODEMAP_DELETE_REPO_MAP,
    #     params={
    #         "root_path": current_path,
    #     },
    # )
    # print("Result: ", result)

    # init repo map
    result = toolset.execute_action(
        action=Action.CODEMAP_INIT_REPO_MAP,
        params={
            "root_path": dir_path,
        },
    )

    print("Result: ", result)

    status = toolset.execute_action(
        action=Action.CODEMAP_GET_REPO_MAP,
        params={
            "root_path": dir_path,
            "target_files": [
                "/Users/soham/composio_sdk/test.py",
                "/Users/soham/composio_sdk/setup.py",
            ],
        },
    )
    print("REPO MAP: ", status)

    # tags = toolset.execute_action(
    #     action=Action.CODEMAP_GENERATE_RANKED_TAGS,
    #     params={
    #         "root_path": dir_path,
    #         "target_files": [
    #             "/Users/soham/composio_sdk/test.py",
    #             "/Users/soham/composio_sdk/setup.py",
    #             "Users/soham/composio_sdk/python/composio/cli/action.py",
    #         ],
    #     },
    # )
    # print("REPO TAGS: ", tags)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
