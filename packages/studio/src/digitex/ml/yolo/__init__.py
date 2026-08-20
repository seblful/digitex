"""The two halves of a YOLO training run.

:mod:`dataset` turns labelled images into the train/val/test tree YOLO reads,
which is file arithmetic a test can drive on a ``tmp_path``. :mod:`training`
hands a config pair to ultralytics and waits, which nothing can do without the
vendor's stack installed. They are separate files because only one of them is
testable.

Import the concrete module, as everywhere else in the project.
"""
