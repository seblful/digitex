"""One module per screen the student can be looking at.

``start`` is first contact and admin approval, ``navigation`` the walk down to a
question set, ``testing`` and ``random`` the two answering loops, ``results``
the screen that closes a session. ``results`` registers no router: that screen
is drawn by the testing loop on its way out, never reached by a tap of its own.
"""
