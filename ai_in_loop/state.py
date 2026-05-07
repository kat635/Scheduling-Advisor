"""State schema for the paper analyzer workflow.

This module defines the ScheduleState TypedDict that flows through the LangGraph
workflow. Each node reads from and writes to fields in this state.
"""

from typing import TypedDict


class ScheduleState(TypedDict, total=False):
    """State object that flows through the scheduling advisor workflow.

    The workflow acts like an advisor, helping students create class
    schedule based on their major requirements, desired classes and
    the hope anchor plan. It uses web scraping to access the course
    catalog and get information on which classes are needed.

    Fields:
    - url: the url of the specific major course catalog (input)
    - result: the output of web scraping. A dict containing title, description and courses (output)
    - error: stores error messages from the handle_errors node (status)
    """

    # === Input ===
    url: str
    result: dict
    error: str
