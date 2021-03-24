"""Another prototype of the State implementation.
Usage
-----
How to import this:
    import streamlit as st
    import st_state_patch
When you do that, you will get 3 new commands in the "st" module:
    * st.State
    * st.SessionState
    * st.GlobalState
The important class here is st.State. The other two are just an alternate API
that provides some syntax sugar.
Using st.State
--------------
Just call st.State() and you'll get a session-specific object to add state into.
To initialize it, just use an "if" block, like this:
    s = st.State()
    if not s:
        # Initialize it here!
        s.foo = "bar"
If you want your state to be global rather than session-specific, pass the
"is_global" keyword argument:
    s = st.State(is_global=True)
    if not s:
        # Initialize it here!
        s.foo = "bar"
Alternate API
-------------
If you think this reads better, you can create session-specific and global State
objects with these commands instread:
    s0 = st.SessionState()
    # Same as st.State()
    s1 = st.GlobalState()
    # Same as st.State(is_global=True)
Multiple states per app
-----------------------
If you'd like to instantiate several State objects in the same app, this will
actually give you 2 different State instances:
    s0 = st.State()
    s1 = st.State()
    print(s0 == s1)  # Prints False
If that's not what you want, you can use the "key" argument to specify which
exact State object you want:
    s0 = st.State(key="user metadata")
    s1 = st.State(key="user metadata")
    print(s0 == s1)  # Prints True
"""

import inspect
import os
import threading
import collections

import streamlit as st

try:
    import streamlit.ReportThread as ReportThread
    from streamlit.server.Server import Server
except Exception:
    # Streamlit >= 0.65.0
    import streamlit.report_thread as ReportThread
    from streamlit.server.server import Server

# Normally we'd use a Streamtit module, but I want a module that doesn't live in
# your current working directory (since local modules get removed in between
# runs), and Streamtit devs are likely to have Streamlit in their cwd.
import sys
GLOBAL_CONTAINER = sys


class State(object):
    def __new__(cls, key=None, is_global=False):
        if is_global:
            states_dict, key_counts = _get_global_state()
        else:
            states_dict, key_counts = _get_session_state()

        if key is None:
            key = _figure_out_key(key_counts)

        if key in states_dict:
            return states_dict[key]

        state = super(State, cls).__new__(cls)
        states_dict[key] = state

        return state

    def __init__(self, key=None, is_global=False):
        pass

    def __bool__(self):
        return bool(len(self.__dict__))

    def __contains__(self, name):
        return name in self.__dict__


def _get_global_state():
    if not hasattr(GLOBAL_CONTAINER, '_global_state'):
        GLOBAL_CONTAINER._global_state = {}
        GLOBAL_CONTAINER._key_counts = collections.defaultdict(int)

    return GLOBAL_CONTAINER._global_state, GLOBAL_CONTAINER._key_counts


def _get_session_state():
    session = _get_session_object()

    curr_thread = threading.current_thread()

    if not hasattr(session, '_session_state'):
        session._session_state = {}

    if not hasattr(curr_thread, '_key_counts'):
        # Put this in the thread because it gets cleared on every run.
        curr_thread._key_counts = collections.defaultdict(int)

    return session._session_state, curr_thread._key_counts


def _get_session_object():
    # Hack to get the session object from Streamlit.

    ctx = ReportThread.get_report_ctx()

    this_session = None
    current_server = Server.get_current()
    if hasattr(current_server, '_session_infos'):
        # Streamlit < 0.56
        session_infos = Server.get_current()._session_infos.values()
    else:
        session_infos = Server.get_current()._session_info_by_id.values()

    for session_info in session_infos:
        s = session_info.session
        if (
            # Streamlit < 0.54.0
            (hasattr(s, '_main_dg') and s._main_dg == ctx.main_dg)
            or
            # Streamlit >= 0.54.0
            (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue)
            or
            # Streamlit >= 0.65.2
            (not hasattr(s, '_main_dg') and s._uploaded_file_mgr == ctx.uploaded_file_mgr)
        ):
            this_session = s

    if this_session is None:
        raise RuntimeError(
            "Oh noes. Couldn't get your Streamlit Session object"
            'Are you doing something fancy with threads?')

    return this_session


def _figure_out_key(key_counts):
    stack = inspect.stack()

    for stack_pos, stack_item in enumerate(stack):
        filename = stack_item[1]
        if filename != __file__:
            break
        else:
            stack_item = None

    if stack_item is None:
        return None

    # Just breaking these out for readability.
    #frame_id = id(stack_item[0])
    filename = stack_item[1]
    # line_no = stack_item[2]
    func_name = stack_item[3]
    # code_context = stack_item[4]

    key = "%s :: %s :: %s" % (filename, func_name, stack_pos)

    count = key_counts[key]
    key_counts[key] += 1

    key = "%s :: %s" % (key, count)

    return key


class SessionState(object):
    def __new__(cls, key=None):
        return State(key=key, is_global=False)


class GlobalState(object):
    def __new__(cls, key=None):
        return State(key=key, is_global=True)


st.State = State
st.GlobalState = GlobalState
st.SessionState = SessionState
