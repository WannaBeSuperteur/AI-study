
from langchain.tools import tool, ToolRuntime
from datetime import date, timedelta


def calculate_date(date_str: str, days: int) -> str:
    """
    Calculate the date before/after N days from original date.
    Create Date : 2026.02.20

    :param date_str: original date string, in the format of "yyyy-mm-dd"
    :param days:     number of days (positive for after, negative for before)
    :param runtime:  LangChain tool runtime
    :return:         the date before/after N days, in the format of "yyyy년 m월 d일"
    """

    original_year = int(date_str.split("-")[0])
    original_month = int(date_str.split("-")[1])
    original_day = int(date_str.split("-")[2])
    original_date = date(original_year, original_month, original_day)

    if days >= 0:
        dest_date = original_date + timedelta(days=days)
    else:
        dest_date = original_date - timedelta(days=days)

    formatted_dest_date = dest_date.strftime('%Y년 %m월 %d일')
    return formatted_dest_date


@tool
def calculate_date_(date_str: str, days: int, runtime: ToolRuntime) -> str:
    """
    Calculate the date before/after N days from original date.
    """

    return calculate_date(date_str, days)


if __name__ == '__main__':
    print(calculate_date("2026-02-01", 15))
