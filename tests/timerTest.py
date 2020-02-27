import timer
import unittest
import time


class Timer_test(unittest.TestCase):
  def testTimer(self):
    with timer.Timer('Timer test') as t:
      time.sleep(0.1)
    with timer.Timer('Timer test', True) as t:
      time.sleep(0.1)
    with timer.Timer('Timer test') as t:
      time.sleep(0.1)

if __name__ == '__main__':
  unittest.main()