from datetime import datetime, timedelta
import numpy as np

class Time:

    def __init__(self, hours, minutes=0):
        self._minutes = minutes
        self._hours = hours
        self._minutes_since_midnight = hours * 60 + minutes

    @property
    def minutes(self):
        return self._minutes

    @property
    def minutes_since_midnight(self):
        return self._minutes_since_midnight


class Task:

    def __init__(self, name, category, duration, start_time=None):
        self.name = name
        self.category = category
        self.start_time = start_time
        self.duration = duration

class TimeBlock:

    def __init__(self, start_min, end_min):
        self._start_min = start_min
        self._end_min = end_min

    @property
    def start_min(self):
        return self._start_min

    @property
    def end_min(self):
        return self._end_min

    @property
    def duration(self):
        return self.end_min - self.start_min

class Slot:

    def __init__(self, start_minute, duration):
        self._start_minute = start_minute
        self._duration = duration

    @property
    def start_time(self):
        return datetime.utcfromtimestamp(self._start_minute * 60)

    @property
    def start_minute(self):
        return self._start_minute

    @property
    def end_time(self):
        return self._start_time + timedelta(minutes=self._duration)

    @property
    def end_minute(self):
        return self.start_minute + self.duration

    @property
    def duration(self):
        return self._duration

class TaskScheduler:

    def __init__(self, tasks, start_hour, end_hour):
        self._tasks = tasks
        self._start_hour = start_hour
        self._end_hour = end_hour

    def add_task(self, task):
        self.tasks.append(task)

    def propose_slots(self, task, slot_step_min=15):
        free_blocks = [TimeBlock(0, (self._end_hour - self._start_hour) * 60)]
        for t in self._tasks:
            sm = t.start_time.minutes_since_midnight - self._start_hour * 60
            em = sm + t.duration
            for b in free_blocks:
                if sm >= b.start_min and em <= b.end_min:
                    new_blocks = free_blocks
                    new_blocks.remove(b)
                    if sm > b.start_min and em < b.end_min:
                        new_blocks.append(TimeBlock(b.start_min, sm))
                        new_blocks.append(TimeBlock(em, b.end_min))
                    elif sm == b.start_min:
                        new_blocks.append(TimeBlock(em, b.end_min))
                    elif em == b.end_min:
                        new_blocks.append(TimeBlock(b.start_min, sm))
                    free_blocks = new_blocks
                    break
        free_blocks = sorted(free_blocks, key=lambda x: x.start_min)

        slots = []
        for b in free_blocks:
            if task.duration <= b.duration:
                sm = b.start_min
                em = sm + task.duration
                while em <= b.end_min:
                    slots.append(Slot(sm + self._start_hour * 60, task.duration))
                    sm += slot_step_min
                    em = sm + task.duration
        return slots

    @property
    def tasks(self):
        return self._tasks

class DiscreteDistribution:

    def __init__(self, frequencies, interval_length=15):
        self._interval_length = interval_length
        interval_starts = [s for s in range(0, 24 * 60, interval_length)]
        for start in interval_starts:
            if start not in frequencies:
                frequencies[start] = 0.0001
        self._frequencies = frequencies
        self.normalize()

    def density_at(self, start_minute):
        return self._frequencies[start_minute]

    def density_for(self, start_minute, end_minute):
        return sum([self.density_at(start) for start in range(start_minute, end_minute + 1, self._interval_length)])

    def normalize(self):
        norm = sum([d for d in self._frequencies.values()])
        self._frequencies = {k:v/norm for k, v in self._frequencies.items()}

    def joint(self, dist):
        frequencies = dict()
        interval_starts = [s for s in range(0, 24 * 60, self._interval_length)]
        for start in interval_starts:
            frequencies[start] = self.density_at(start) * dist.density_at(start)
        return DiscreteDistribution(frequencies, self._interval_length)

    @property
    def interval_length(self):
        return self._interval_length


class UniformDistribution(DiscreteDistribution):

    def __init__(self, start_time, end_time, interval_length=15):

        start_m = start_time.minutes_since_midnight
        end_m = end_time.minutes_since_midnight

        duration = end_m - start_m
        interval_count = duration / interval_length
        density_per_interval = 1.0 / interval_count

        frequencies = {start: density_per_interval for start in range(start_m, end_m + 1, interval_length)}
        DiscreteDistribution.__init__(self, frequencies, interval_length)


from abc import ABCMeta, abstractmethod

class LikelihoodEstimator(metaclass=ABCMeta):

    def __init__(self, priors):
        self.priors = priors

    @abstractmethod
    def likelihood(self, task, priors):
        pass

class ProbabilisticTaskScheduler:

    def __init__(self, task_scheduler, likelihood_estimator):
        self._task_scheduler = task_scheduler
        self._likelihood_estimator = likelihood_estimator

    def propose_slots(self, task, slot_step_min=15):
        slots = self._task_scheduler.propose_slots(task, slot_step_min)
        if not slots:
            return []
        prior = self._likelihood_estimator.likelihood(task, self._likelihood_estimator.priors)

        slot_densities = [(prior.density_for(s.start_minute, s.end_minute), s) for s in slots]
        norm = sum([d[0] for d in slot_densities])
        return [(d[0] / norm, d[1]) for d in slot_densities]

    def choose_slot(self, task, slot_step_min=15, sample_count=1):
        slot_pairs = self.propose_slots(task, slot_step_min)
        if not slot_pairs:
            return None
        samples = np.random.multinomial(sample_count, list(zip(*slot_pairs))[0])
        pair_idx = np.argmax(samples)
        return slot_pairs[pair_idx][1]

    def choose_slots(self, tasks, slot_step_min=15, sample_count=1):
        tasks = sorted(tasks, key=lambda t: t.duration, reverse=True)
        slots = []
        original_tasks = self._task_scheduler.tasks
        for t in tasks:
            slot = self.choose_slot(t, slot_step_min, sample_count)
            if not slot:
                continue
            t.start_time = Time(slot.start_minute / 60, slot.start_minute % 60)
            self._task_scheduler.add_task(t)
            slots.append(slot)
        self._task_scheduler._tasks = original_tasks
        return slots

    @property
    def tasks(self):
        return self._task_scheduler.tasks
