import pytest
import numpy.testing
from datetime import datetime

from scheduler.scheduler import ProbabilisticTaskScheduler, TaskScheduler, Task, Slot, Time, LikelihoodEstimator, UniformDistribution

CATEGORY_PRIORS = dict(wellness=UniformDistribution(Time(hours=14), Time(hours=20)),
              learning=UniformDistribution(Time(hours=10), Time(hours=14)),
              work=UniformDistribution(Time(hours=11), Time(hours=15)),
              personal=UniformDistribution(Time(hours=8), Time(hours=22)),
              fun=UniformDistribution(Time(hours=12), Time(hours=22)),
              chores=UniformDistribution(Time(hours=16), Time(hours=22)))


class TaskCategoryLikelihoodEstimator(LikelihoodEstimator):

    def likelihood(self, task, priors):
        if not task.category in priors:
            return UniformDistribution(Time(hours=0), Time(hours=24))
        return priors[task.category]

@pytest.yield_fixture(autouse=True)
def scheduler():
    t1 = Task("Workout in the park", "wellness", 60, Time(hours=11))
    t2 = Task("Read a book", "learning", 30, Time(hours=10, minutes=30))
    yield ProbabilisticTaskScheduler(TaskScheduler([t1, t2], 10, 12), likelihood_estimator=TaskCategoryLikelihoodEstimator(CATEGORY_PRIORS))

def test_propose_all_slots(scheduler):
    t3 = Task("Meditate", "wellness", 15)
    slots = scheduler.propose_slots(t3)
    assert slots
    assert len(slots) == 2

def test_proposed_slots_should_have_probability(scheduler):
    t3 = Task("Meditate", "wellness", 15)
    slots = scheduler.propose_slots(t3)
    assert not isinstance(slots[0], (float, Slot))

def test_choose_slot_should_return_single_slot(scheduler):
    t3 = Task("Meditate", "wellness", 15)
    slot = scheduler.choose_slot(t3)
    assert isinstance(slot, Slot)

def test_proposed_slots_should_have_correct_probabilities(scheduler):
    t3 = Task("Meditate", "wellness", 15)
    slots = scheduler.propose_slots(t3)
    p1, p2 = slots[0][0], slots[1][0]
    numpy.testing.assert_almost_equal(p1, p2)

def test_chosen_slot_should_have_correct_start_minute(scheduler):
    t3 = Task("Meditate", "wellness", 15)
    slot = scheduler.choose_slot(t3)
    assert slot.start_minute == 600 or slot.start_minute == 615

def test_choose_slots_for_multiple_tasks(scheduler):
    t1 = Task("Workout in the park", "wellness", 60, Time(hours=11))
    t2 = Task("Read a book", "learning", 30, Time(hours=10, minutes=30))
    scheduler = ProbabilisticTaskScheduler(TaskScheduler([t1, t2], 8, 12), likelihood_estimator=TaskCategoryLikelihoodEstimator(CATEGORY_PRIORS))
    t3 = Task("Meditate", "wellness", 15)
    t4 = Task("Read a paper", "learning", 30)
    slots = scheduler.choose_slots([t3, t4])
    assert len(slots) == 2

def test_schedule_larger_tasks_first():
    t1 = Task("Workout in the park", "wellness", 60, Time(hours=11))
    t2 = Task("Read a book", "learning", 30, Time(hours=10, minutes=30))
    scheduler = ProbabilisticTaskScheduler(TaskScheduler([t1, t2], 8, 12), likelihood_estimator=TaskCategoryLikelihoodEstimator(CATEGORY_PRIORS))
    t3 = Task("Meditate", "wellness", 15)
    t4 = Task("Read a paper", "learning", 30)
    slots = scheduler.choose_slots([t3, t4])
    assert slots[0].duration > slots[1].duration

def test_none_slot_when_no_time():
    t1 = Task("Workout in the park", "wellness", 60, Time(hours=11))
    t2 = Task("Read a book", "learning", 30, Time(hours=10, minutes=30))
    scheduler = ProbabilisticTaskScheduler(TaskScheduler([t1, t2], 10, 12), likelihood_estimator=TaskCategoryLikelihoodEstimator(CATEGORY_PRIORS))
    t3 = Task("Meditate", "wellness", 60)
    slot = scheduler.choose_slot(t3)
    assert slot is None

def test_empty_slots_when_no_time():
    t1 = Task("Workout in the park", "wellness", 60, Time(hours=11))
    t2 = Task("Read a book", "learning", 60, Time(hours=10))
    scheduler = ProbabilisticTaskScheduler(TaskScheduler([t1, t2], 10, 12), likelihood_estimator=TaskCategoryLikelihoodEstimator(CATEGORY_PRIORS))
    t3 = Task("Meditate", "wellness", 15)
    t4 = Task("Read a paper", "learning", 30)
    slots = scheduler.choose_slots([t3, t4])
    assert len(slots) == 0

def test_assign_slot_for_available_time():
    t1 = Task("Workout in the park", "wellness", 60, Time(hours=11))
    t2 = Task("Read a book", "learning", 30, Time(hours=10))
    scheduler = ProbabilisticTaskScheduler(TaskScheduler([t1, t2], 10, 12), likelihood_estimator=TaskCategoryLikelihoodEstimator(CATEGORY_PRIORS))
    t3 = Task("Meditate", "wellness", 15)
    t4 = Task("Read a paper", "learning", 60)
    slots = scheduler.choose_slots([t3, t4])
    assert len(slots) == 1
