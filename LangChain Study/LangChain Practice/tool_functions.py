
from langchain.tools import tool, ToolRuntime
from datetime import date, timedelta


def calculate_date(date_str: str, days: int) -> str:
    """
    Calculate the date before/after N days from original date.
    Create Date : 2026.02.20

    :param date_str: original date string, in the format of "yyyy-mm-dd"
    :param days:     number of days (positive for after, negative for before)
    :return:         the date before/after N days, in the format of "yyyy년 m월 d일"
    """

    original_year = int(date_str.split("-")[0])
    original_month = int(date_str.split("-")[1])
    original_day = int(date_str.split("-")[2])
    original_date = date(original_year, original_month, original_day)

    dest_date = original_date - timedelta(days=days)
    formatted_dest_date = dest_date.strftime('%Y년 %m월 %d일')
    return formatted_dest_date


@tool
def calculate_date_(date_str: str, days: int, runtime: ToolRuntime) -> str:
    """
    Calculate the date before/after N days from original date.
    """

    return calculate_date(date_str, days)


def calculate_day_of_week(date_str: str) -> str:
    """
    Calculate day-of-week of the date.
    Create Date : 2026.02.20

    :param date_str: original date string, in the format of "yyyy-mm-dd"
    :return:         day-of-week of the date, in the format of "O요일"
    """

    year = int(date_str.split("-")[0])
    month = int(date_str.split("-")[1])
    day = int(date_str.split("-")[2])
    original_date = date(year, month, day)

    dow = original_date.weekday()
    dow_mapping = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    dow_name = dow_mapping[dow]

    return dow_name


@tool
def calculate_day_of_week_(date_str: str, runtime: ToolRuntime) -> str:
    """
    Calculate day-of-week of the date.
    """

    return calculate_day_of_week(date_str)


if __name__ == '__main__':
    print(calculate_date("2026-02-01", 15))
    print(calculate_date("2026-02-01", -15))
    print(calculate_date("2026-02-01", 365))
    print(calculate_date("2026-02-01", -365))
    print(calculate_date("2026-02-01", 0))

    print(calculate_day_of_week("2026-02-20"))
    print(calculate_day_of_week("2026-03-01"))
    print(calculate_day_of_week("2028-01-01"))
