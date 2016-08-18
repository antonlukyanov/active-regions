from datetime import datetime


class Timer:
    """
    Простой таймер для замеров времени выполнения кода.
    """

    def __init__(self):
        self.start = datetime.now()

    def show(self):
        print('Running time: %s' % (datetime.now() - self.start))
