
from langchain.tools import tool
from langgraph.store.memory import InMemoryStore

USER_INFO_KEY = "user_info"


@tool
def get_user_info(info_type: str, store: InMemoryStore) -> str:
    """
    Get user info of info_type.
    Create Date : 2026.02.20
    Last Update Date : 2026.02.26 (do not use Runtime, only use store)

    :param info_type: info key to search
    :return:          matching value for the info key
    """

    user_id = 'tester'
    namespace = ("users", user_id)
    memory = store.get(namespace, USER_INFO_KEY)

    if memory:
        try:
            return f"{info_type}: {memory.value[info_type]}"
        except:
            return f"{info_type} 정보 없음"
    else:
        return f"{info_type} 정보 없음"


@tool
def set_user_info(info_type: str, info_value: str, store: InMemoryStore):
    """
    Store user info of info_type as info_value.
    Create Date : 2026.02.20
    Last Update Date : 2026.02.26 (do not use Runtime, only use store)

    :param info_type:  info key to store info value
    :param info_value: info value to be stored
    :return:           matching value for the info key
    """

    user_id = 'tester'
    namespace = ("users", user_id)
    memory = store.get(namespace, USER_INFO_KEY)

    if memory:
        user_info = memory.value
    else:
        user_info = {}

    user_info[info_type] = info_value
    store.put(namespace, USER_INFO_KEY, user_info)


# 모든 정보 조회
def get_all_user_info(store, user_id) -> dict:
    """
    Show all stored user info.

    :param store:   user info store
    :param user_id: user ID
    :return:        (key, value) pair of all stored user info
    """

    # search user_info (dict-like)
    namespace = ("users", user_id)
    memory = store.get(namespace, USER_INFO_KEY)
    return memory.value
