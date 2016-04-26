import pytest
from datetime import datetime

from scheduler.scheduler import TaskScheduler, Task, Time

@pytest.yield_fixture(autouse=True)
def scheduler():
    t1 = Task("Workout in the park", "wellness", 60, Time(hours=11))
    t2 = Task("Read a book", "learning", 30, Time(hours=10, minutes=30))
    yield TaskScheduler([t1, t2], 10, 12)

def test_add_tasks(scheduler):
    assert len(scheduler.tasks) == 2

def test_propose_slots_at_start(scheduler):
    t3 = Task("Meditate", "wellness", 15)
    slots = scheduler.propose_slots(t3)
    assert slots
    assert len(slots) == 2
    assert slots[0].start_time.hour == 10
    assert slots[0].start_time.minute == 0
    assert slots[1].start_time.hour == 10
    assert slots[1].start_time.minute == 15

def test_propose_slots_in_between_tasks(scheduler):
    t1 = Task("Workout in the park", "wellness", 60, Time(hours=11))
    t2 = Task("Read a book", "learning", 30, Time(hours=10))
    s = TaskScheduler([t1, t2], 10, 12)
    t3 = Task("Meditate", "wellness", 15)
    slots = s.propose_slots(t3)
    assert slots
    assert len(slots) == 2

def test_propose_slots_at_end(scheduler):
    t1 = Task("Workout in the park", "wellness", 60, Time(hours=10, minutes=30))
    t2 = Task("Read a book", "learning", 30, Time(hours=10, minutes=0))
    s = TaskScheduler([t1, t2], 10, 12)
    t3 = Task("Meditate", "wellness", 15)
    slots = s.propose_slots(t3)
    assert slots
    assert len(slots) == 2


def test_no_slots_for_full_schedule(scheduler):
    t1 = Task("Workout in the park", "wellness", 60, Time(hours=11))
    t2 = Task("Read a book", "learning", 60, Time(hours=10))
    s = TaskScheduler([t1, t2], 10, 12)
    t3 = Task("Meditate", "wellness", 15)
    slots = s.propose_slots(t3)
    assert not slots

def test_slots_for_longer_task(scheduler):
    t1 = Task("Workout in the park", "wellness", 60, Time(hours=10, minutes=30))
    t2 = Task("Read a book", "learning", 30, Time(hours=10))
    s = TaskScheduler([t1, t2], 10, 12)
    t3 = Task("Meditate", "wellness", 30)
    slots = s.propose_slots(t3)
    assert slots
    assert len(slots) == 1

