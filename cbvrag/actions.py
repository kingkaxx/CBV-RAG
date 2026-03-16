from enum import IntEnum


ACTION_ENUM_VERSION = "cbvrag_action_enum_v1"


class Action(IntEnum):
    ANSWER_DIRECT = 0
    RETRIEVE_MORE_SMALL = 1
    RETRIEVE_MORE_LARGE = 2
    SPAWN_COUNTERFACTUAL = 3
    SELECT_CONTEXT = 4
    VERIFY_CHEAP = 5
    VERIFY_LLM = 6
    PRUNE_BRANCH = 7
    MERGE_BRANCHES = 8
    SUMMARIZE_STATE = 9
    STOP_AND_ANSWER = 10


def action_names() -> list[str]:
    return [a.name for a in Action]
