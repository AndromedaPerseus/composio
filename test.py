from composio import ComposioToolSet, Action, App

toolset = ComposioToolSet()
import os

# get current path
current_path = os.getcwd()

result = toolset.execute_action(
    action=Action.CODEINDEX_CREATE_INDEX,
    params={
        "dir_to_index_path": current_path,
        "force_index": True,
    },
)

print("Result: ", result)

status = toolset.execute_action(
    action=Action.CODEINDEX_INDEX_STATUS,
    params={
        "dir_to_index_path": current_path,
    },
)
print("Status: ", status)

import time

# # wait for 2 minutes
time.sleep(120)
